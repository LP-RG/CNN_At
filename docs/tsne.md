# t-SNE Visualization

The t-SNE tool runs [scikit-learn t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
on CNN feature activations (or raw pixels), overlays classification errors, and
produces both static PNG outputs and an interactive Dash web application.

---

## 1 — CLI Usage (`res_net_training.py`)

The t-SNE analysis is triggered by passing `--tsne` alongside the usual model arguments.

### Minimal example

```bash
python res_net_training.py \
    --model_name lenet5 \
    --tsne
```

This runs t-SNE on the **exact** (float) checkpoint with all defaults (see table below).

### Full argument reference

| Flag | Type | Default | Description |
|---|---|---|---|
| `--tsne` | flag | — | **Required** to activate t-SNE mode. |
| `--model_name` | str | `resnet` | Model key: `lenet5`, `resnet` (=resnet20), `resnet8`, `vgg16`, `alexnet_cifar10`, `resnet56`. |
| `--tsne_perplexity` | int | `30` | t-SNE perplexity. Lower values focus on local structure. |
| `--tsne_max_iter` | int | `1000` | Maximum t-SNE iterations. |
| `--tsne_max_train` | int | `2000` | Max training samples fed into t-SNE. |
| `--tsne_max_test` | int | `1000` | Max test samples fed into t-SNE. |
| `--tsne_seed` | int | `42` | RNG seed for subsampling and t-SNE init. |
| `--tsne_classes` | int… | `None` (all) | Restrict to a subset of classes, e.g. `--tsne_classes 0 1 2`. |
| `--tsne_feature_space` | `layer`/`pixels` | `layer` | Feature type to embed (see §2). |
| `--tsne_feature_layer` | str… | `penultimate` | Layer alias(es) or explicit module path(s) (see §2). You can specify multiple layers to process them all simultaneously. |
| `--tsne_stages` | str… | `exact` | Which checkpoints to visualise: `exact`, `quantized`, `approximate`. |
| `--tsne_multiplier_path` | str… | `None` | Path(s) to `.npy` lookup table(s); **required** for `approximate` stage. Accepts multiple paths to run several approximate models simultaneously. |
| `--bit_width` | int | `8` | Bit width for quantized/approximate checkpoint filenames. |
| `--show_misclassifications` | flag | `False` | Also save a grid of misclassified test images as a PNG. |
| `--tsne-no-save-static` | flag | — | Suppress static PNG output. |
| `--tsne-no-save-dash-artifact` | flag | — | Suppress `.npz` Dash artifact output. |

### Multi-stage comparison example

```bash
python res_net_training.py \
    --model_name resnet \
    --tsne \
    --tsne_stages exact quantized approximate \
    --tsne_multiplier_path multipliers/my_table.npy multipliers/my_other_table.npy \
    --tsne_feature_layer penultimate \
    --tsne_max_train 3000 --tsne_max_test 1500 \
    --tsne_seed 7
```

When multiple stages are specified, the **same** random train/test subsets are
reused across all stages, making the embeddings directly comparable.

---

## 2 — Feature Spaces and Layer Aliases

The `--tsne_feature_space` parameter controls what gets embedded:

| `feature_space` | What is embedded |
|---|---|
| `"layer"` | Activations extracted via a **forward hook** on the named module specified by `--tsne_feature_layer`. |
| `"pixels"` | Flattened raw input tensors (no model inference for training data). |

### Layer aliases (for `--tsne_feature_layer`)

You can pass one or more of these aliases separated by spaces. All requested layers will be processed and saved into the same run directory.

| Alias | Resolves to |
|---|---|
| `penultimate` | Second-to-last `nn.Linear`; or last `AdaptiveAvgPool2d` for ResNet-style models. |
| `logits` | Last `nn.Linear` (raw class scores before softmax). |
| `conv1` | First `Conv2d_custom` in the model. |
| `conv2` | Second `Conv2d_custom` (or first if only one exists). |
| `block1` | First of `layer1` / `block1` / `pool1` found in `named_modules()`. |
| `block2` | First of `layer2` / `block2` / `pool2` found in `named_modules()`. |
| *any other string* | Used **verbatim** as a `named_modules()` path, e.g. `layer2.0` or `features.3`. |

> If the literal string is already a valid `named_modules()` key, it is used
> directly without alias lookup.

---

## 3 — Output Files

All outputs are written under `./plots` (configurable via `save_dir` in the Python API). Each run is logically grouped into a timestamped directory containing a `metadata.json` file.

```
plots/
└── <feature_space>/          # "layer" or "pixels"
    └── <model_name>/
        └── run_YYYYMMDD_HHMMSS/
            ├── metadata.json
            ├── tsne/
            │   └── tsne_<model>[_layer-<layer>][_<tag>][_classes<ids>].png
            ├── misclassified/
            │   └── misclassified_<model>[…].png
            └── dash_data/
                └── tsne_<model>[_layer-<layer>][_<tag>][_classes<ids>].npz
```

### Static PNG (`tsne/`)
A matplotlib scatter plot with:
- **Grey dots** — training samples (position only, not evaluated).
- **Coloured dots** — test samples, coloured by true class label.
- **Red ×** — misclassified test samples.
- **Class label text** — centred on each cluster.
- Accuracy annotation in the lower-left corner.

### Misclassification grid (`misclassified/`)
Saved only when `--show_misclassifications` is set. Each cell shows the raw
image with `true=` / `pred=` labels in red.

### Dash artifact (`.npz`)
A compressed NumPy archive consumed by the interactive app (see [Dash App](dash_app.md)).
Contains all arrays needed to reconstruct the plot and preview misclassified
images without re-running t-SNE.

### Run Metadata (`metadata.json`)
Automatically generated on every run. Logs hyper-parameters (seed, max_train, etc.) ensuring that artifacts in the same `run_` directory are completely comparable.

---

## 4 — Common Workflows

### A — Quick sanity check (raw pixels, all classes)

```bash
python res_net_training.py \
    --model_name lenet5 \
    --tsne \
    --tsne_feature_space pixels \
    --tsne_max_train 1000 --tsne_max_test 500
```

### B — Layer embedding, compare exact vs. quantized

```bash
python res_net_training.py \
    --model_name resnet \
    --tsne \
    --tsne_stages exact quantized \
    --tsne_feature_layer penultimate
```

### C — Multi-layer comparison

```bash
python res_net_training.py \
    --model_name lenet5 \
    --tsne \
    --tsne_stages exact \
    --tsne_feature_layer conv1 penultimate
```

### D — Approximate hardware, specific classes, with image grid

```bash
python res_net_training.py \
    --model_name resnet \
    --tsne \
    --tsne_stages approximate \
    --tsne_multiplier_path multipliers/my_table.npy \
    --tsne_classes 3 5 8 \
    --show_misclassifications \
    --tsne_feature_layer conv2
```

### E — View an existing run in the Dash app

```bash
python3 apps/tsne_dash_app.py
# In the browser: paste a run directory path, e.g. plots/layer/resnet/run_20260515_181919
```
