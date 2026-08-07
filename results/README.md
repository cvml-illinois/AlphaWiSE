# Results

Nothing in this directory is committed; it is where artifact runs write their outputs.
The repository distributes the pipeline, not numbers: regenerate any table or figure and
compare the produced values against the published paper directly.

Each artifact writes under its own subdirectory, for example:

```
results/main_comparison/   curves CSV plus the rendered table (csv, md, tex)
results/retention_curves/     retention curves (pdf, png)
results/smoke/    everything the synthetic end-to-end run produces
```

The commands that populate these are declared per artifact in
[`configs/artifacts/`](../configs/artifacts) and driven by:

```bash
python -m alphawise.reproduce --list-artifacts
python -m alphawise.reproduce --artifact main_comparison --verify
```

`--verify` checks that every stage produced the files it declares. The default output
root is `./results`; override it with `ALPHAWISE_RESULTS`.
