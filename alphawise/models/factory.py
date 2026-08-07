"""The single supported way to construct the backbone.

AudioCLIP's constructor takes roughly twenty arguments, and getting the vision
arguments wrong silently yields a ModifiedResNet backbone with a different
parameter set instead of the ViT-B/32 one the paper uses. Every call site in this
repository therefore goes through :func:`build_audioclip`.

The audio branch always uses GroupNorm (``layernorm_audio=True``). BatchNorm is not
an option: its running statistics have no meaningful semantics across continual
phases, and interpolating affine parameters whose paired statistics differ between
two checkpoints is not well defined.
"""

from .audioclip import AudioCLIP
from ..utils.paths import asset

EMBED_DIM = 512
VISION_PATCH_SIZE = 32
VISION_WIDTH = 768
VISION_LAYERS = 12

#: Interpolation coefficients, one per parameter tensor. Asserted at fusion time.
NUM_COEFFICIENTS = 499
COEFFICIENTS_PER_BRANCH = {"image": 152, "text": 149, "audio": 195, "temperature": 3}


def build_audioclip(checkpoint=None) -> AudioCLIP:
    """ViT-B/32 image and text branches, ESResNeXt-FBSP audio branch, GroupNorm audio.

    ``checkpoint`` is loaded strictly, and only after the audio branch has been
    converted to GroupNorm, so a mismatch surfaces as an error rather than as
    silently untouched weights.
    """
    return AudioCLIP(
        embed_dim=EMBED_DIM,
        vision_patch_size=VISION_PATCH_SIZE,
        vision_width=VISION_WIDTH,
        vision_layers=VISION_LAYERS,
        multilabel=False,
        pretrained=str(checkpoint) if checkpoint is not None else False,
        clip_model_pth=str(asset("ViT_CLIP.pt")),
        layernorm_audio=True,
    )


def branch_of(key: str) -> str:
    """Which encoder a checkpoint key belongs to, for coefficient bookkeeping."""
    if key.startswith("visual"):
        return "image"
    if key.startswith("audio"):
        return "audio"
    if key.startswith("logit_scale"):
        return "temperature"
    return "text"
