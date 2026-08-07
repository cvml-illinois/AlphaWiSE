import os

import torch
import torch.nn.functional as F
import torch.nn as nn

from .clip import CLIP
from .clip.clip import tokenize
from .esresnet import ESResNeXtFBSP
from ..utils.paths import asset

from typing import List
from typing import Tuple
from typing import Union
from typing import Optional

ClipFeatures = Tuple[
    Optional[torch.Tensor],  # audio
    Optional[torch.Tensor],  # image
    Optional[torch.Tensor]   # audio
]


ClipLogits = Tuple[
    Optional[torch.Tensor],  # audio x image
    Optional[torch.Tensor],  # audio x text
    Optional[torch.Tensor]   # image x text
]


ClipOutput = Tuple[
    Tuple[ClipFeatures, ClipLogits],
    Optional[torch.Tensor]   # loss
]

def bn_to_ln(module: nn.Module):
    for name, child in module.named_children():
        # descend first so we catch BNs inside Sequential blocks
        bn_to_ln(child)

        if isinstance(child, nn.BatchNorm2d):
            ln = nn.GroupNorm(1, child.num_features, eps=child.eps, affine=True)

            # copy affine parameters so you don't lose the learned scale/shift
            with torch.no_grad():
                ln.weight.copy_(child.weight)
                ln.bias.copy_(child.bias)

            setattr(module, name, ln)          # swap in-place

class AudioCLIP(CLIP):

    def __init__(self,
                 embed_dim: int = 1024,
                 # vision
                 image_resolution: int = 224,
                 vision_layers: Union[Tuple[int, int, int, int], int] = (3, 4, 6, 3),
                 vision_width: int = 64,
                 vision_patch_size: Optional[int] = None,
                 # text
                 context_length: int = 77,
                 vocab_size: int = 49408,
                 transformer_width: int = 512,
                 transformer_heads: int = 8,
                 transformer_layers: int = 12,
                 # audio
                 n_fft: int = 2048,
                 hop_length: Optional[int] = 561,
                 win_length: Optional[int] = 1654,
                 window: Optional[str] = 'blackmanharris',
                 normalized: bool = True,
                 onesided: bool = True,
                 spec_height: int = -1,
                 spec_width: int = -1,
                 apply_attention: bool = True,
                 multilabel: bool = True,
                 pretrained: Union[bool, str] = True,
                 clip_model_pth = None,
                 layernorm_audio = False):

        super(AudioCLIP, self).__init__(
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            context_length=context_length,
            vocab_size=vocab_size,
            transformer_width=transformer_width,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers
        )

        _model_dir = os.path.dirname(os.path.abspath(__file__))
        self.audio = ESResNeXtFBSP(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            normalized=normalized,
            onesided=onesided,
            spec_height=spec_height,
            spec_width=spec_width,
            num_classes=embed_dim,
            apply_attention=apply_attention,
            pretrained=str(asset('ESRNXFBSP.pt'))
        )

        # Given an argument switch batch norm in audio transformer with 
        if layernorm_audio:
            bn_to_ln(self.audio)
            print(self.audio)

        print("Loaded pretrained ESResNEXT weights for audio head (From Independent Training)...")

        self.multilabel = multilabel
        self.pretrained = pretrained

        self.logit_scale_ai = torch.nn.Parameter(torch.log(torch.ones([]) * 100))
        self.logit_scale_at = torch.nn.Parameter(torch.log(torch.ones([]) * 100))
        
        if isinstance(self.pretrained, str):
            my_weights = torch.load(self.pretrained, map_location='cpu')
            if "model_state_dict" not in my_weights.keys():
                self.load_state_dict(my_weights, strict=True)
            else:
                my_weights["model_state_dict"] = { (k[5:] if k.startswith('base.') else k): v
                     for k, v in my_weights["model_state_dict"].items() }
                self.load_state_dict(my_weights["model_state_dict"], strict=True)
                print("Loading Trained Weights ... ")
        elif self.pretrained:
            print("Doen't every reach here (will error)")
            ## Doen't every reach here (will error)
            self.load_state_dict(torch.load(
                asset('CLIP.pt'),
                map_location='cpu'
            ), strict=False)
            print('Image & Text weights loaded')
            # try:
            # Pretraining of AudioSet (Always supposed to this)
            self.audio.load_state_dict(torch.load(
                asset('ESRNXFBSP.pt'),
                map_location='cpu'
            ), strict=False)
            # except RuntimeError as ex:
            #     print(ex)
            #     print('Audio weights loaded')
        else:
            if clip_model_pth:
                print("Loading Full ViT Clip weights (for Visual and Text)")
                self.load_state_dict(torch.load(
                    clip_model_pth,
                    map_location='cpu'
                ), strict=False)
            else:
                self.load_state_dict(torch.load(
                    asset('CLIP.pt'),
                    map_location='cpu'
                ), strict=False)
            print('Image & Text weights loaded')
            print("You need to make scripts to cooperatively train the Audio Head")

        self.embed_dim = embed_dim

    @property
    def device(self):
        return self.visual.conv1.weight.device

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        return self.audio(audio.to(self.device))

    def encode_text(self,
                    text: List[List[str]],
                    base_str: str = '{}',
                    batch_indices: Optional[torch.Tensor] = None) -> torch.Tensor:

        if batch_indices is not None:
            text = [text[idx] for idx in batch_indices]

        text_joined = [', '.join(entities) for entities in text]
        text_tokens = torch.cat([
            tokenize(base_str.format(entities)) for entities in text_joined
        ])
        text_tokens = text_tokens.to(self.device)

        return super(AudioCLIP, self).encode_text(text_tokens)

    def forward(self,
                audio: Optional[torch.Tensor] = None,
                image: Optional[torch.Tensor] = None,
                text: Optional[List[List[str]]] = None,
                batch_indices: Optional[torch.Tensor] = None) -> ClipOutput:

        audio_features = None
        image_features = None
        text_features = None
        sample_weights = None

        if audio is not None:
            audio_features = self.encode_audio(audio)
            audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)

        if image is not None:
            image_features = self.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if text is not None:
            if batch_indices is None:
                batch_indices = torch.arange(len(text), dtype=torch.int64, device=self.device)

            text_features = self.encode_text(text, '{}', batch_indices)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            if hasattr(self, 'class_weights') and hasattr(self, 'label_to_class_idx'):
                sample_weights = torch.stack([
                    sum(self.class_weights[self.label_to_class_idx[label]] for label in entities)
                    for idx, entities in enumerate(text) if idx in batch_indices
                ])

        features: ClipFeatures = (audio_features, image_features, text_features)

        logit_scale_ai = torch.clamp(self.logit_scale_ai.exp(), min=1.0, max=100.0)
        logit_scale_at = torch.clamp(self.logit_scale_at.exp(), min=1.0, max=100.0)
        logit_scale_it = torch.clamp(self.logit_scale.exp(), min=1.0, max=100.0)

        logits_audio_image = None
        logits_audio_text = None
        logits_image_text = None

        if (audio_features is not None) and (image_features is not None):
            logits_audio_image = logit_scale_ai * audio_features @ image_features.T

        if (audio_features is not None) and (text_features is not None):
            logits_audio_text = logit_scale_at * audio_features @ text_features.T

        if (image_features is not None) and (text_features is not None):
            logits_image_text = logit_scale_it * image_features @ text_features.T

        logits: ClipLogits = (logits_audio_image, logits_audio_text, logits_image_text)

        loss = self.loss_fn(logits, sample_weights)
        if audio is not None and loss is not None:
            loss = loss + self.audio.loss_ttf(self.device)

        return (features, logits), loss

    def loss_fn(self, logits: ClipLogits, sample_weights: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        logits_audio_image, logits_audio_text, logits_image_text = logits

        if logits_audio_image is not None:
            batch_size = logits_audio_image.shape[0]
        elif logits_audio_text is not None:
            batch_size = logits_audio_text.shape[0]
        elif logits_image_text is not None:
            batch_size = logits_image_text.shape[0]
        else:
            return None

        reference = torch.arange(
            batch_size,
            dtype=torch.int64,
            device=self.device
        )

        loss = torch.tensor(0.0, dtype=self.dtype, device=self.device)

        num_modalities: int = 0
        scale = torch.tensor(1.0, dtype=self.dtype, device=self.device)

        if logits_audio_image is not None:
            loss_ai = F.cross_entropy(
                logits_audio_image, reference, weight=sample_weights
            ) + F.cross_entropy(
                logits_audio_image.transpose(-1, -2), reference, weight=sample_weights
            )
            loss = loss + loss_ai
            num_modalities += 1

        if logits_audio_text is not None:
            loss_at = F.cross_entropy(
                logits_audio_text, reference, weight=sample_weights
            ) + F.cross_entropy(
                logits_audio_text.transpose(-1, -2), reference, weight=sample_weights
            )
            loss = loss + loss_at
            num_modalities += 1

        if logits_image_text is not None:
            loss_it = F.cross_entropy(
                logits_image_text, reference, weight=sample_weights
            ) + F.cross_entropy(
                logits_image_text.transpose(-1, -2), reference, weight=sample_weights
            )
            loss = loss + loss_it
            num_modalities += 1

        for idx in range(num_modalities):
            scale = scale * (idx + 1)

        return loss / scale

    @property
    def loss_fn_name(self) -> str:
        return 'Cross Entropy'

if __name__ == "__main__":
    ac = AudioCLIP()