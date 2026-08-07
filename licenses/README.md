# Third-party code and assets

This repository vendors source from three upstream projects rather than depending on them as
packages, because each needed small modifications (most importantly replacing BatchNorm with
GroupNorm in the audio branch, which is what makes weight-space interpolation well defined).
Vendoring keeps those modifications visible in the diff instead of hidden in a patch step.

Modifications are listed per entry. Everything else is upstream code, unchanged.

| Path | Upstream | License | Modifications |
|---|---|---|---|
| `alphawise/models/clip/` | [OpenAI CLIP](https://github.com/openai/CLIP) | MIT | Import paths rewritten for this package. Asset loading routed through `alphawise.utils.paths`. |
| `alphawise/utils/simple_tokenizer.py` | [OpenAI CLIP](https://github.com/openai/CLIP) | MIT | Vocabulary path resolved lazily through `alphawise.utils.paths` instead of a fixed relative path. |
| `alphawise/models/esresnet/` | [AudioCLIP](https://github.com/AndreyGuzhov/AudioCLIP) (ESResNeXt-FBSP audio encoder) | MIT | Import paths rewritten. Abstract base class import inlined, see below. |
| `alphawise/models/audioclip.py` | [AudioCLIP](https://github.com/AndreyGuzhov/AudioCLIP) | MIT | `bn_to_ln` GroupNorm conversion of the audio branch; per-pair temperature parameters; asset loading routed through `alphawise.utils.paths`. |
| `alphawise/utils/transforms.py` | [AudioCLIP](https://github.com/AndreyGuzhov/AudioCLIP) | MIT | Abstract base class import inlined, see below. |
| `alphawise/_interfaces.py` | `ignite_trainer` (the AudioCLIP training harness) | MIT | Copied verbatim. Only two abstract base classes are used; importing the upstream package pulled in pytorch-ignite and visdom for nothing, so they are inlined here to keep this repository self-contained. |

## Pretrained weights

The backbone initialisations are not redistributed here. `python scripts/fetch_assets.py` obtains
them and verifies their checksums.

| Asset | Origin | Used for |
|---|---|---|
| `ViT_CLIP.pt` | OpenAI CLIP ViT-B/32 | Image and text branch initialisation |
| `ESRNXFBSP.pt` | AudioCLIP ESResNeXt-FBSP, AudioSet-pretrained | Audio branch initialisation |
| `bpe_simple_vocab_16e6.txt.gz` | OpenAI CLIP | Text tokenizer vocabulary |

## Dataset

AudioSet is distributed by Google as metadata referencing YouTube videos, under CC BY 4.0 for the
annotations. The audio and video are not redistributable, so this repository ships the segment
manifests that define the benchmark splits and a downloader, never media. See `docs/data.md`.
