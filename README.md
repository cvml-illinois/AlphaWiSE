# AlphaWiSE: Adaptive Weight Interpolation for Continual Multimodal Representation Learning

This repository contains the official code for the paper *AlphaWiSE: Adaptive Weight
Interpolation for Continual Multimodal Representation Learning*.

**TLDR:** a continual-learning checkpoint commits every retrieval direction to one
stability-plasticity trade-off. Interpolating two frozen checkpoints with one learned
coefficient per parameter tensor recovers a better model than either, at no extra
inference cost.

## Approach

AlphaWiSE fuses two frozen AudioCLIP checkpoints trained over the same phase sequence: an
unconstrained endpoint from sequential fine-tuning, and a stability endpoint from EWC, LwF
or iCaRL. For each of the 499 corresponding parameter tensors it learns one coefficient,

    theta_p = alpha_p * theta_un_p + (1 - alpha_p) * theta_reg_p,    alpha_p = sigmoid(beta_p)

and optimizes only the 499 betas, on the 840-exemplar memory plus a class-balanced draw
from the current phase (the fitting-set sizes and the reasoning are recorded in
[configs/fusion/_base.yaml](configs/fusion/_base.yaml)). The fused tensors are then
written into a single standalone checkpoint with the same architecture, the same
parameter count, and the same inference cost as either source.

```python
from alphawise.fusion.core import TensorWiseInterpolation

model = TensorWiseInterpolation(
    endpoint_un="checkpoints/normal_forward/phase_7/model_weights_epoch_30.pth",
    endpoint_reg="checkpoints/ewc_forward/phase_7/model_weights_epoch_30.pth",
    init_alpha=0.5,
)
# 499 coefficients: 152 image, 149 text, 195 audio, 3 temperature. Both endpoints frozen,
# so backward() reaches model.beta and nothing else.
loss.backward()
torch.save(model.materialize(), "fused.pth")
```

Holding every coefficient at 0.5 instead of fitting it gives the WiSE-FT baseline, which is
why `alphawise/fusion/wiseft.py` is a thin wrapper over the same code rather than a second
implementation. See [alphawise/fusion/core.py](alphawise/fusion/core.py) and
[alphawise/fusion/fit.py](alphawise/fusion/fit.py).

### Applying the pipeline to your own model

The interpolation itself has no AudioCLIP knowledge. `blend` works on any two
state-dict-compatible checkpoints of any architecture, one coefficient per tensor you
name, and passes unnamed entries (buffers) through untouched:

```python
from alphawise.fusion.blend import blend

fused_sd = blend(plastic_sd, stable_sd, alphas)   # alpha * plastic + (1 - alpha) * stable
```

To fit the coefficients instead of choosing them, mirror
[alphawise/fusion/fit.py](alphawise/fusion/fit.py): wrap `blend` so the coefficients are
the only `nn.Parameter`, run your model functionally over the blended state dict, and
backpropagate your task loss into the coefficient logits. Everything specific to this
paper (the memory construction, the retrieval objective, the phase sequence) sits outside
that seam. One architectural caveat transfers: interpolate normalisation layers whose
statistics live in the affine parameters (GroupNorm, LayerNorm), not BatchNorm running
statistics.

## Installation

The code requires `python>=3.11`, `torch>=2.6`, and a `torchvision` built against the same
torch release (`>=0.21` is the pairing for torch 2.6). NumPy is capped below 2.5, and a
newer NumPy shadowing the one torchvision was compiled against fails at import with
`operator torchvision::nms does not exist`. Every floor is behavioural; the reason for
each is recorded next to it in [pyproject.toml](pyproject.toml).

```bash
cd AlphaWiSE
pip install -e .
python scripts/fetch_assets.py
```

`fetch_assets` verifies the four backbone files in `assets/` against the SHA-256 digests
embedded in the script, and downloads whatever is missing when
`ALPHAWISE_ASSET_BASE_URL` names a mirror; no download location is hardcoded. To use a
copy you already hold, set `ALPHAWISE_ASSETS` to that directory instead.
[assets/README.md](assets/README.md) lists each file, where it comes from, and its
digest.

## Quick start

The whole pipeline runs on a generated corpus with the same on-disk shape as AudioSet, so
nothing has to be downloaded first:

```bash
make smoke
```

This writes a synthetic mini-corpus, trains a short phase sequence, fits the coefficients,
materializes the fused checkpoint, evaluates it, and renders a table in the shape of
Table 1. It takes about nine minutes on CPU alone, and needs no GPU, no AudioSet and no
environment variables; its only prerequisite is the backbone assets from the installation
step. Its numbers are at chance by construction, since one epoch over a generated corpus
learns nothing; what it demonstrates is that the stages connect.

## Data

AudioSet is distributed as metadata referencing YouTube videos. The media are not
redistributable and clips are removed over time, so the corpus itself is never shipped:
you assemble a copy once and point the pipeline at it. [data/README.md](data/README.md)
documents the on-disk layout the loaders parse, the segment CSVs and ontology they
expect, and the environment variables that name an existing copy.

The benchmark is deterministic given a copy: the class-to-phase partition, the per-phase
candidate pools and the exemplar selections are all derived in code from the ontology and
the segment CSVs, not from shipped lists. A copy that retrieves fewer clips scores below
the published numbers for reasons unrelated to the method, which is why evaluation
refuses to run on fewer than 7000 eval samples rather than quietly reporting on a
shrunken pool.

## Checkpoints

No weights are distributed with this repository: every checkpoint is produced by running
the pipeline. [checkpoints/README.md](checkpoints/README.md) documents the layout
training writes and evaluation expects, and
[configs/checkpoints/registry_840.yaml](configs/checkpoints/registry_840.yaml) maps every
method key to the directory its run is expected in: externally trained trajectories under
`ALPHAWISE_EXTERNAL_CHECKPOINTS`, anything this repository trains itself under
`ALPHAWISE_CHECKPOINTS`. Each artifact config declares the inputs its commands need, and
the driver reports precisely what is missing rather than failing mid-run.

## Benchmark protocol

| Phase | Classes | Role |
|---|---|---|
| 0 | 79 | Shared frozen bootstrap, never trained by the continual pipeline |
| 1 to 7 | 79 each | Trained in sequence; earlier data is reachable only through the exemplar memory |

Evaluation is directed retrieval on audio-to-text, image-to-audio and image-to-text,
reported as R@1 and mAP over a per-phase 79-class candidate pool, averaged over the phases
seen so far and at the last phase. The audio branch uses GroupNorm rather than BatchNorm,
so interpolated affine parameters are never paired with mismatched running statistics.

## Usage

Train one continual-learning trajectory. `--method` loads `configs/method/ewc.yaml` as
the last config overlay, so one flag selects both the implementation and its published
hyperparameters:

```bash
python -m alphawise.train \
  --method ewc \
  --class-order forward \
  --phases 1,2,3,4,5,6,7
```

Fit the interpolation coefficients between two finished trajectories:

```bash
python -m alphawise.fusion.fit \
  --config configs/fusion/normal_ewc.yaml \
  --out-dir checkpoints/alphawise_n_ewc_forward \
  --phases 1,2,3,4,5,6,7
```

Evaluate registered checkpoints over the phase sequence:

```bash
python -m alphawise.eval.runner \
  --config configs/checkpoints/registry_840.yaml \
  --method alphawise_n_ewc \
  --class-order forward \
  --out results/runs/alphawise_n_ewc.csv
```

## Evaluating and comparing against the paper

Each artifact below runs the full evaluation protocol behind one result of the paper:
train or mount the checkpoints it names, run it, and you get the same tables and figures
computed over your runs. That is the intended workflow for comparison: swap your own
method or backbone into the registry and these commands score it under exactly the
protocol the published baselines were scored under, rather than a reimplementation of it.

```bash
python -m alphawise.reproduce --list-artifacts
python -m alphawise.reproduce --artifact main_comparison --verify
```

| Artifact | Protocol it runs | Command |
|---|---|---|
| `main_comparison` | Main comparison: 11 methods, 3 retrieval directions, phase-averaged and last-phase | `python -m alphawise.reproduce --artifact main_comparison` |
| `objective_ablation` | Coefficient-objective study: one fused pair refitted under 4 objectives | `python -m alphawise.reproduce --artifact objective_ablation` |
| `transfer_metrics` | Backward and forward transfer over the full phase-by-phase matrix | `python -m alphawise.reproduce --artifact transfer_metrics` |
| `retention_curves` | Retention curves per phase and direction | `python -m alphawise.reproduce --artifact retention_curves` |
| `embedding_tsne` | Embedding structure, base methods against their fused counterparts | `python -m alphawise.reproduce --artifact embedding_tsne` |
| `coefficient_profile` | Learned coefficient profile by depth and parameter type | `python -m alphawise.reproduce --artifact coefficient_profile` |

The repository ships the pipeline, not numbers or weights, so `--verify` is structural:
after the steps it checks that every stage produced the files it declares, and before any
compute it reports every declared input that is absent on your machine. Compare the
regenerated tables against the published paper directly; each artifact config carries
provenance notes stating exactly how its published numbers were produced, including the
settings that no surviving log records. Every resolved command is printed, so the wrapper
can be bypassed entirely and any step rerun by hand.

Figures 1 and 2 are hand-authored diagrams with no generator, so they are not artifacts
here.

## Repository structure

```
alphawise/
  models/       AudioCLIP backbone; factory.py is the only supported constructor
  data/         phase datasets, the exemplar memory, the fusion fitting set, the synthetic corpus
  train/        one training loop and one module per continual-learning method
  fusion/       blend.py and core.py blend, fit.py fits the coefficients, wiseft.py fixes them
  eval/         directed retrieval metrics, the phase transfer matrix, BWT and FWT
  analysis/     renderers for the machine-generated tables and figures of the paper
assets/         backbone initialisations (not committed; see its README for files and digests)
checkpoints/    training and fusion outputs (not committed; its README documents the layout)
configs/        one config per method, per fusion variant, and per paper artifact
data/           the synthetic corpus or your AudioSet copy (not committed; layout in its README)
results/        artifact outputs (not committed; its README says what lands where)
tests/          smoke.sh, the end-to-end pipeline check on synthetic data
```

## Notes on reproducibility

Training sets no RNG seed, so retraining does not reproduce the released checkpoints bit
for bit, and verification uses tolerance bands rather than equality. Passing `--seed` makes
a run deterministic and changes nothing when it is omitted.

Fusion is deterministic given its endpoints: blending at a fixed coefficient reproduces a
released fused checkpoint to a maximum absolute deviation of 0 across all 499 tensors.


## License

MIT, see [LICENSE](LICENSE). Vendored upstream code and the pretrained initialisations
carry their own terms, listed in [licenses/README.md](licenses/README.md).
