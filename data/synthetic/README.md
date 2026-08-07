# Synthetic corpus

Generated, never committed. `make smoke` (or the command below) writes a tiny corpus here
with the same on-disk shape as AudioSet: `train/` and `eval/` splits, each with `images/`,
`audio/`, a `segments.csv`, and a shared `ontology.json`.

```bash
python -m alphawise.data.synthetic --out data/synthetic --classes 16 --per-class 6 --seed 42
```

Deterministic under its seed (the same seed writes byte-identical files), with disjoint
train and eval clip ids, so a non-chance retrieval number on it would mean leakage. It is
regenerated on every smoke run; delete this directory freely. See
[../README.md](../README.md) for the real AudioSet layout the same shape mirrors.
