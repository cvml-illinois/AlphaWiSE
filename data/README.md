# Data layout

Nothing in this directory is committed. It holds either the synthetic smoke corpus, which
is generated on demand, or your copy of AudioSet, which you assemble once. Every path
below can also live elsewhere and be pointed at through environment variables, so nothing
needs to be inside the repository tree.

## Synthetic corpus (no download, no GPU)

The smoke pipeline generates a tiny corpus with the same on-disk shape as AudioSet and
runs the whole pipeline on it:

```bash
python -m alphawise.data.synthetic --out data/synthetic
# or simply: make smoke
```

Generation is deterministic under its seed (42 by default), and the train and eval halves
draw disjoint clip sets.

## Real AudioSet

AudioSet is distributed as YouTube segment ids; you download the media yourself. The
pipeline reads one directory of extracted frames (jpg) and one of audio clips (wav) per
split, plus the segment CSVs and the ontology:

```
<data root>/
├── train/
│   ├── img/                            frames, named <video id>*.jpg
│   ├── wav/                            clips, named Y<video id>*.wav
│   └── unbalanced_train_segments.csv   the official AudioSet segment list
├── eval/
│   ├── eval_img/
│   ├── eval_wav/
│   └── eval_segments.csv
└── ontology/ontology.json              the official AudioSet ontology
```

The default root is `./data`; override the root or any single location:

```bash
export ALPHAWISE_DATA_ROOT=/path/to/audioset
# or per location:
export AUDIOSET_TRAIN_IMG=... AUDIOSET_TRAIN_WAV=... AUDIOSET_TRAIN_CSV=...
export AUDIOSET_EVAL_IMG=...  AUDIOSET_EVAL_WAV=...  AUDIOSET_EVAL_CSV=...
export AUDIOSET_ONT=...
```

### Wav filename convention

A clip's YouTube id is recovered from its filename. The published runs used two parser
modes, `strip_y` for train wavs (drops a leading `Y`) and `plain` for eval wavs; both are
kept as the defaults because the published numbers depend on them, and a corrected `dual`
mode is available through `data.wav_id_mode`. The module docstring in
[`alphawise/data/audioset.py`](../alphawise/data/audioset.py) explains the fork and its
consequences precisely.

### Phase splits

The 7-phase class sequence and the class ordering (`class_order: forward | reverse`) are
declared in [`configs/base.yaml`](../configs/base.yaml), not by files in this directory:
the split is derived from the ontology and the declared ordering at load time, so there
is no split file to download.
