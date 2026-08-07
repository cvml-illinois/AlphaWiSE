# Checkpoint layout

Nothing in this directory is committed and no weights are distributed with this
repository: every checkpoint is produced by running the pipeline. This file records the
layout the code writes and expects, so a tree produced on one machine can be mounted on
another.

## Where checkpoints live

Three roots, all overridable by environment variable (defaults in
[`configs/base.yaml`](../configs/base.yaml)):

```bash
export ALPHAWISE_CHECKPOINTS=...           # runs this repository trains (default ./checkpoints)
export ALPHAWISE_EXTERNAL_CHECKPOINTS=...  # runs trained elsewhere and mounted read only
export ALPHAWISE_BOOTSTRAP=...             # the shared phase 0 models (default <checkpoints>/bootstrap)
```

## What training writes

```
<checkpoint root>/<method>_<class_order>/phase_<p>/model_weights_epoch_<e>.pth
```

for example `checkpoints/ewc_forward/phase_3/model_weights_epoch_30.pth`. The continual
protocol needs the shared bootstrap in place before phase 1:

```
<bootstrap root>/<class_order>/phase_0/model_weights_epoch_30.pth
```

Training refuses to start phase 1 without it, because every method must leave the same
phase 0 model for the comparison to be fair.

## What fusion writes

Per phase and endpoint pair, into the fused run's directory:

- `learnable_wiseft_model.pth`: the materialized fused checkpoint, same architecture and
  size as either endpoint.
- The learned coefficient file holding the 499 pre-sigmoid logits (`beta`) and their
  sigmoided values, self describing with the endpoint pair and objective.

## How evaluation finds runs

[`configs/checkpoints/registry_840.yaml`](../configs/checkpoints/registry_840.yaml) maps
each method key to its run directory and phase list. `filename: null` means "take the
highest numbered `model_weights_epoch_<N>.pth` present". Add or override entries there,
not in code, when your directory names differ.
