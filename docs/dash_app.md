# Interactive Dash App (`apps/tsne_dash_app.py`)

> **Status:** Work in progress — this section will be expanded as the app develops.

The Dash app provides an interactive browser-based viewer for t-SNE artifacts
produced by the t-SNE tool (see [t-SNE Visualization](tsne.md)).

---

## Launching

```bash
python apps/tsne_dash_app.py [--host 0.0.0.0] [--port 8050] [--debug]
```

Then open `http://localhost:8050` in your browser.

| Flag | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Host address to bind to. |
| `--port` | `8050` | Port to listen on. |
| `--debug` | `False` | Enable Dash debug/hot-reload mode. |

---

## Usage

1. Paste the path to a `.npz` artifact (relative to the project root, or absolute)
   into the **"Artifact path"** field.
2. Click **"Load Artifact"** — the interactive t-SNE scatter appears.
3. **Click a red ×** (misclassified point) to see the raw image and metadata
   (true label, predicted label, test index) in the right panel.
4. Click **"Clear / Reset View"** to unload the current artifact.

> **Note:** Only red × misclassified crosses are interactive.
> Clicking grey or coloured dots shows a message but no image preview.

---

## Planned

- [ ] On-demand t-SNE re-run from within the app (no CLI required)
- [ ] Side-by-side stage comparison (exact / quantized / approximate)

