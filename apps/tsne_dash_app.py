import argparse
import base64
import os
import sys
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Ensure project root is importable when running apps/tsne_dash_app.py directly.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from dash import Dash, Input, Output, State, callback_context, dcc, html
from dash.exceptions import PreventUpdate
from PIL import Image
from modules.tsne_utils import (
    get_misclassified_indices,
    load_dash_artifact,
)


def _image_to_data_url(flat_img, image_shape):
    """Convert a flattened sample into a base64 PNG data URL for html.Img."""
    c, h, w = image_shape
    img = flat_img.reshape(c, h, w).astype(np.float32, copy=False)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    # Match previous matplotlib gray_r visual behavior.
    if c == 1:
        pil_img = Image.fromarray(255 - img_u8[0], mode="L")
    else:
        pil_img = Image.fromarray(np.transpose(img_u8, (1, 2, 0)), mode="RGB")

    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_plot_figure(data):
    """Build the interactive Plotly t-SNE figure and misclassification mapping."""
    import plotly.graph_objects as go
    import seaborn as sns

    X_2d = data["X_2d"]
    y_all = data["y_all"]
    test_mask = data["test_mask"].astype(bool)
    y_test_sub = data["y_test_sub"]
    y_pred_sub = data["y_pred_sub"]

    train_mask = ~test_mask
    wrong_mask = np.zeros(len(y_all), dtype=bool)
    wrong_mask[test_mask] = (y_test_sub != y_pred_sub)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=X_2d[train_mask, 0],
        y=X_2d[train_mask, 1],
        mode="markers",
        marker=dict(size=5, color="#cccccc", opacity=0.4),
        name="Train (not evaluated)",
        hoverinfo="skip",
        showlegend=True,
    ))

    unique_labels = np.unique(y_all)
    palette = np.array(sns.color_palette("hls", len(unique_labels)))
    label_to_color = {
        int(lab): f"rgb({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)})"
        for lab, rgb in zip(unique_labels, palette)
    }

    for lab in unique_labels:
        mask = test_mask & (y_all == lab)
        if not np.any(mask):
            continue
        fig.add_trace(go.Scattergl(
            x=X_2d[mask, 0],
            y=X_2d[mask, 1],
            mode="markers",
            marker=dict(size=8, color=label_to_color[int(lab)], opacity=0.85),
            name=f"Test label {int(lab)}",
            customdata=np.full(mask.sum(), -1, dtype=np.int64),
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>true=%{text}<extra></extra>",
            text=np.full(mask.sum(), int(lab), dtype=np.int64),
            showlegend=True,
        ))

    wrong_indices = np.where(wrong_mask)[0]
    wrong_test_indices = get_misclassified_indices(y_test_sub, y_pred_sub)
    mis_trace_idx = None
    if len(wrong_indices) > 0:
        mis_trace_idx = len(fig.data)
        fig.add_trace(go.Scattergl(
            x=X_2d[wrong_indices, 0],
            y=X_2d[wrong_indices, 1],
            mode="markers",
            marker=dict(symbol="x", size=14, color="red", line=dict(width=2, color="red")),
            name=f"Misclassified ({len(wrong_indices)})",
            customdata=wrong_test_indices,
            hovertemplate=("x=%{x:.2f}<br>y=%{y:.2f}<br>"
                           "true=%{text}<br>pred=%{meta}<extra></extra>"),
            text=y_test_sub[wrong_test_indices],
            meta=y_pred_sub[wrong_test_indices],
            showlegend=True,
        ))

    ref_mask = test_mask if np.any(test_mask) else np.ones(len(y_all), dtype=bool)
    for lab in unique_labels:
        pts = X_2d[ref_mask & (y_all == lab)]
        if len(pts) == 0:
            continue
        xtext, ytext = np.median(pts, axis=0)
        fig.add_annotation(
            x=float(xtext),
            y=float(ytext),
            text=str(int(lab)),
            showarrow=False,
            font=dict(size=20, color="black"),
            bgcolor="rgba(255,255,255,0.8)",
        )

    title_value = data.get("title", "t-SNE")
    # Backward/forward compatible: artifacts may expose title as numpy scalar
    # (with .item()) or already as a plain Python string.
    if hasattr(title_value, "item"):
        title_value = title_value.item()
    title = str(title_value)
    fig.update_layout(
        title=title,
        template="simple_white",
        width=1000,
        height=800,
        clickmode="event+select",
        legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
        margin=dict(l=20, r=20, t=80, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
    )
    return fig, mis_trace_idx, wrong_test_indices


def _blank_figure(message="Load an artifact to visualize t-SNE."):
    """Return a placeholder figure shown before any artifact is loaded."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(
        template="simple_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=40, b=20),
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=18, color="#666"),
            )
        ],
    )
    return fig


def _load_artifact(artifact_path):
    """Load an artifact and build both figure and lightweight callback state."""
    data = load_dash_artifact(artifact_path)
    fig, mis_trace_idx, _ = _build_plot_figure(data)
    payload = {
        "path": artifact_path,
        "mis_trace_idx": mis_trace_idx,
    }
    return fig, payload


def create_app():
    app = Dash(__name__)
    initial_msg = "Load an artifact, then click a red cross."
    app.layout = html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "340px minmax(700px, 1fr)",
            "gap": "12px",
            "padding": "10px",
            "height": "100vh",
            "boxSizing": "border-box",
            "overflow": "hidden",
        },
        children=[
            html.Div(
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "8px",
                    "overflowY": "auto",
                    "paddingRight": "6px",
                },
                children=[
                    html.H3("t-SNE Dashboard", style={"margin": "0 0 8px 0"}),
                    html.Label("Artifact path"),
                    dcc.Input(id="artifact-path", type="text", style={"width": "100%"},
                              placeholder="plots/layer/lenet5/dash_data/....npz"),
                    html.Button("Load Artifact", id="load-artifact", n_clicks=0),
                    html.Button("Clear / Reset View", id="clear-reset", n_clicks=0),
                    dcc.Loading(
                        type="dot",
                        children=[
                            html.Pre(
                                id="status-log",
                                children="Idle. Load an artifact to start.",
                                style={
                                    "whiteSpace": "pre-wrap",
                                    "overflowWrap": "anywhere",
                                    "wordBreak": "break-word",
                                    "fontSize": "11px",
                                    "maxHeight": "180px",
                                    "overflowY": "auto",
                                },
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "minmax(520px, 1fr) 320px",
                    "gap": "12px",
                    "height": "100%",
                    "minWidth": 0,
                },
                children=[
                    html.Div(
                        style={"display": "flex", "flexDirection": "column", "minWidth": 0},
                        children=[
                            dcc.Loading(
                                type="default",
                                children=[
                                    dcc.Graph(
                                        id="tsne-graph",
                                        figure=_blank_figure(),
                                        style={"height": "calc(100vh - 44px)"},
                                        config={"displaylogo": False},
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        style={
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "8px",
                            "height": "calc(100vh - 44px)",
                            "overflowY": "auto",
                            "padding": "8px",
                            "border": "1px solid #ddd",
                            "borderRadius": "6px",
                            "backgroundColor": "#fafafa",
                        },
                        children=[
                            dcc.Loading(
                                type="circle",
                                children=[
                                    html.H4("Misclassified Preview", style={"margin": "0"}),
                                    html.Div(id="preview-message", children=initial_msg),
                                    html.Img(
                                        id="preview-image",
                                        style={
                                            "width": "100%",
                                            "maxHeight": "320px",
                                            "objectFit": "contain",
                                            "border": "1px solid #ddd",
                                            "backgroundColor": "white",
                                        },
                                    ),
                                    html.Pre(
                                        id="preview-meta",
                                        style={
                                            "whiteSpace": "pre-wrap",
                                            "margin": "0",
                                            "fontSize": "12px",
                                            "overflowX": "auto",
                                        },
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dcc.Store(id="artifact-state"),
                ],
            ),
        ],
    )

    # load artifact or reset view
    @app.callback(
        Output("tsne-graph", "figure"),
        Output("artifact-state", "data"),
        Output("status-log", "children"),
        Output("artifact-path", "value"),
        Input("load-artifact", "n_clicks"),
        Input("clear-reset", "n_clicks"),
        State("artifact-path", "value"),
        prevent_initial_call=True,
    )
    def _load_or_reset(load_clicks, clear_clicks, artifact_path):
        trigger = callback_context.triggered_id
        try:
            if trigger == "clear-reset":
                return _blank_figure(), None, "Idle. Load an artifact to start.", ""
            if trigger == "load-artifact":
                if not artifact_path:
                    raise ValueError("Please provide an artifact path to load.")
                full_path = str((ROOT / artifact_path).resolve()) if not os.path.isabs(artifact_path) else artifact_path
                fig, payload = _load_artifact(full_path)
                return fig, payload, f"Loaded artifact:\n{full_path}", artifact_path
        except Exception as exc:
            return {}, None, f"Error:\n{exc}", artifact_path or ""
        raise PreventUpdate

    # click-to-preview misclassification
    @app.callback(
        Output("preview-message", "children"),
        Output("preview-image", "src"),
        Output("preview-meta", "children"),
        Input("clear-reset", "n_clicks"),
        Input("tsne-graph", "clickData"),
        State("artifact-state", "data"),
    )
    def _update_preview(clear_clicks, click_data, state):
        if callback_context.triggered_id == "clear-reset":
            return initial_msg, "", ""
        if not state:
            return "Load an artifact first.", "", ""
        if not click_data:
            return "Click a red misclassified cross to preview image.", "", ""

        mis_trace_idx = state["mis_trace_idx"]
        if mis_trace_idx is None:
            return "No misclassified points in this artifact.", "", ""

        point = click_data["points"][0]
        curve = int(point.get("curveNumber", -1))
        if curve != mis_trace_idx:
            return "Only red misclassified crosses are interactive.", "", ""

        pi = int(point.get("pointIndex", -1))
        artifact = load_dash_artifact(state["path"])
        wrong_test_indices = get_misclassified_indices(
            artifact["y_test_sub"], artifact["y_pred_sub"]
        )
        if pi < 0 or pi >= len(wrong_test_indices):
            return "Could not resolve clicked point.", "", ""

        test_idx = int(wrong_test_indices[pi])
        image_shape = artifact["image_shape"]
        X_test_pixels = artifact["X_test_pixels"]
        y_test_sub = artifact["y_test_sub"]
        y_pred_sub = artifact["y_pred_sub"]
        img_src = _image_to_data_url(X_test_pixels[test_idx], image_shape)
        meta = (
            f"artifact={state['path']}\n"
            f"test_idx={test_idx}\n"
            f"true={int(y_test_sub[test_idx])}\n"
            f"pred={int(y_pred_sub[test_idx])}"
        )
        return "Selected misclassified sample:", img_src, meta

    return app


def main():
    parser = argparse.ArgumentParser(description="Dash app for t-SNE artifacts and on-demand runs.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
