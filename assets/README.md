# Backbone initialisations

These files are not in the repository. They are large, they are redistributed by their
upstream projects under their own terms, and they never change, so shipping them here
would only add weight and license surface. See [licenses/README.md](../licenses/README.md).

Place the four files in this directory, or point the resolver at a directory that already
holds them:

```bash
export ALPHAWISE_ASSETS=/path/to/your/weights
```

Then verify them:

```bash
python scripts/fetch_assets.py
```

The tool checks each file against the SHA-256 digests embedded in the script, so a
truncated or substituted file is caught rather than loaded. If `ALPHAWISE_ASSET_BASE_URL`
is set, anything missing is first fetched from `<base url>/<name>`; nothing is downloaded
otherwise, and no download location is hardcoded.

| File | Origin | Used for | SHA-256 |
|---|---|---|---|
| `ViT_CLIP.pt` | OpenAI CLIP ViT-B/32 checkpoint | Image and text branch initialisation | `9e468423360d2afa152d16f2176c8cc928e0014ffd65c695f2ec66a17b7e8fbb` |
| `CLIP.pt` | OpenAI CLIP full model, as shipped with the AudioCLIP release | AudioCLIP construction | `124e6d59d54c0837c456c953b3147adf438a21570d5c5b01ad83f6ad3e78d15e` |
| `ESRNXFBSP.pt` | AudioCLIP ESResNeXt-FBSP release | Audio branch initialisation | `acfcc6b0a0f07025cc660ed6dcf9b5a05fbd7c8ebf65edaa0a81d88a92ff6834` |
| `bpe_simple_vocab_16e6.txt.gz` | OpenAI CLIP repository | Text tokenizer vocabulary | `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a` |

The CLIP files come from the [OpenAI CLIP](https://github.com/openai/CLIP) project and the
audio weights from the [AudioCLIP](https://github.com/AndreyGuzhov/AudioCLIP) release
assets; obtain them from those projects under their terms.

Anything missing raises at construction time naming the file and this command, rather than
failing later inside a state-dict load.

## What is not here

Trained checkpoints, meaning the per-phase continual-learning endpoints and the fused
models, are not distributed at all. [checkpoints/README.md](../checkpoints/README.md)
describes the layout the pipeline expects when you train your own.
