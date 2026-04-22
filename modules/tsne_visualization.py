import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import torch
from sklearn.manifold import TSNE


def _filter_loader(loader, classes):
    """Return a new DataLoader restricted to the given class labels."""
    targets = np.array(loader.dataset.targets)
    keep_idx = np.where(np.isin(targets, classes))[0]
    # Wraps the original dataset to only include the indices of the classes we want
    # stores two things:
    # 1. reference to the original dataset
    # 2. indices of the classes we want
    subset = torch.utils.data.Subset(loader.dataset, keep_idx)
    # Wraps the subset dataset into a DataLoader
    return torch.utils.data.DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
    )


def _subsample_loader(loader, n_max, rng):
    """Return a new DataLoader with at most n_max randomly selected samples."""
    n = min(n_max, len(loader.dataset))
    idx = rng.choice(len(loader.dataset), n, replace=False)
    subset = torch.utils.data.Subset(loader.dataset, idx)
    return torch.utils.data.DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
    )


def _collect_inputs(loader):
    """Collect raw inputs and labels from a DataLoader without running the model."""
    inputs_list, labels_list = [], []
    for inputs, labels in loader:
        inputs_list.append(inputs.cpu())
        labels_list.append(labels.cpu())
    # concatenate inputs and convert to numpy array
    X = torch.cat(inputs_list).numpy() 
    y = torch.cat(labels_list).numpy()
    # reshape to (n, -1) where n is the number of samples and 
    # -1 is the automatic dimension determined by the shape of the input
    return X.reshape(len(X), -1), y

def _collect_predictions(model, loader, device):
    """Forward-pass over a DataLoader; return (X_flat, y_true, y_pred)."""
    # use learned stats (not batch stats) for BatchNorm
    model.eval()
    inputs_list, labels_list, preds_list = [], [], []
    # no gradients saves GPU memory
    # During a normal forward pass PyTorch builds a computation graph 
    # it records every operation so it can run backpropagation later, no need
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)     # (batch, n_classes) logits
            _, preds = outputs.max(1)   # finds the maximum along dimension 1, to get predicted class
            inputs_list.append(inputs.cpu())
            labels_list.append(labels.cpu())
            preds_list.append(preds.cpu())

    X = torch.cat(inputs_list).numpy()
    y = torch.cat(labels_list).numpy()
    y_pred = torch.cat(preds_list).numpy()
    return X.reshape(len(X), -1), y, y_pred    # flatten images to (n, 1024)

#TODO: Check for way to iterate trough the layers of the arbitrary models
#--------------------------------------------------
def _collect_fc1_features(model, loader, device, collect_preds=False):
    """Collect fc1 activations via forward hook(lenet5 only); optionally return predictions."""
    if not hasattr(model, "fc1"):
        raise ValueError("Model has no 'fc1' layer for feature-hook extraction.")

    model.eval()
    features_list, labels_list, preds_list = [], [], []

    def _hook(_module, _inputs, output):
        # output shape for LeNet5 fc1: (batch, 84)
        features_list.append(output.detach().cpu())

    handle = model.fc1.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                labels_list.append(labels.cpu())
                if collect_preds:
                    _, preds = outputs.max(1)
                    preds_list.append(preds.cpu())
    finally:
        handle.remove()

    X_feat = torch.cat(features_list).numpy()
    y = torch.cat(labels_list).numpy()
    if collect_preds:
        y_pred = torch.cat(preds_list).numpy()
        return X_feat, y, y_pred
    return X_feat, y

#--------------------------------------------------
def plot_tsne_embedding_cnn(X_2d, y, title=None, y_pred=None, test_mask=None,
                            save_path=None):
    """
    t-SNE scatter with CNN misclassification overlay.
    Plots t-SNE embedding with CNN misclassification overlay.
    """
    X_2d = np.asarray(X_2d)
    y = np.asarray(y)

    unique_labels = np.unique(y)
    n_classes = len(unique_labels)
    palette = np.array(sns.color_palette("hls", n_classes))
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    color_indices = np.array([label_to_idx[int(lab)] for lab in y])

    f, ax = plt.subplots(figsize=(10, 8))

    if test_mask is not None:
        test_mask = np.asarray(test_mask, dtype=bool)
        train_mask = ~test_mask

        ax.scatter(X_2d[train_mask, 0], X_2d[train_mask, 1],
                   lw=0, s=15, c="#cccccc", alpha=0.4, zorder=1,
                   label="Train (not evaluated)")
        ax.scatter(X_2d[test_mask, 0], X_2d[test_mask, 1],
                   lw=0, s=50, c=palette[color_indices[test_mask]],
                   alpha=0.85, zorder=2, label="Test (true label)")
    else:
        ax.scatter(X_2d[:, 0], X_2d[:, 1],
                   lw=0, s=40, c=palette[color_indices],
                   alpha=0.7, zorder=2)

    if y_pred is not None:
        # y_pred covers only the evaluated subset:
        #   - the test split (len = test_mask.sum()) when test_mask is given
        #   - all points otherwise
        y_pred = np.asarray(y_pred)
        eval_mask = test_mask if test_mask is not None else np.ones(len(y), dtype=bool)
        y_eval = y[eval_mask]

        wrong_in_eval = y_eval != y_pred
        wrong = np.zeros(len(y), dtype=bool)
        wrong[eval_mask] = wrong_in_eval

        if wrong.sum() > 0:
            ax.scatter(X_2d[wrong, 0], X_2d[wrong, 1],
                       marker="x", s=120, c="red", linewidths=1.8,
                       label=f"Misclassified ({wrong.sum()})", zorder=5)

        acc = (y_eval == y_pred).mean()
        ax.annotate(
            f"CNN accuracy: {acc:.1%}",
            xy=(0.02, 0.02), xycoords="axes fraction", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax.axis("off")
    ax.axis("tight")

    ref_mask = (test_mask if test_mask is not None
                else np.ones(len(y), dtype=bool))
    for lab in unique_labels:
        pts = X_2d[ref_mask & (y == lab)]
        if len(pts) == 0:
            continue
        xtext, ytext = np.median(pts, axis=0)
        txt = ax.text(xtext, ytext, str(lab), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal(),
        ])

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        f.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.close(f)
    return f, ax
    
def show_misclassified_images(X, y, y_pred, image_shape, save_path,
                              classes=None, n_cols=8):
    """Display a grid of CNN-misclassified test images."""
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)

    wrong = y != y_pred
    if classes is not None:
        classes = np.asarray(classes)
        wrong = wrong & np.isin(y, classes)

    indices = np.where(wrong)[0]
    if len(indices) == 0:
        print("No misclassified images to show.")
        return

    n = len(indices)
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))
    c, h, w = image_shape

    classes_str = f" (classes {classes.tolist()})" if classes is not None else ""
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.4, n_rows * 1.8 + 0.4))
    axes = np.array(axes).reshape(-1)

    for ax_i, idx in enumerate(indices):
        img = X[idx].reshape(c, h, w)
        # normalize to [0, 1] for display
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        if c == 1:
            axes[ax_i].imshow(img[0], cmap="gray_r", interpolation="nearest")
        else:
            # expects (H, W, C) but we have (C, H, W)
            axes[ax_i].imshow(img.transpose(1, 2, 0), interpolation="nearest")
        axes[ax_i].set_title(f"#{idx}\ntrue={y[idx]}\npred={y_pred[idx]}",
                             fontsize=7, color="red")
        axes[ax_i].axis("off")

    for ax_i in range(n, len(axes)):
        axes[ax_i].axis("off")

    fig.suptitle(f"CNN misclassifications{classes_str} — {n} samples", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Misclassified images saved to {save_path}")
    plt.close(fig)

def run_tsne_cnn_experiment(model, train_loader, test_loader, device,
                             model_name="lenet5", perplexity=30,
                             max_iter=1000, max_train=2000, max_test=1000,
                             save_dir="./plots", classes=None, seed=42,
                             show_misclassifications=False, image_shape=None,
                             feature_space="fc1"):
    """Run t-SNE with CNN misclassification overlay."""
    rng = np.random.default_rng(seed)

    if classes is not None:
        classes = np.asarray(classes)
        train_loader = _filter_loader(train_loader, classes)
        test_loader  = _filter_loader(test_loader,  classes)
        print(f"Filtered loaders to classes {classes.tolist()}: "
              f"{len(train_loader.dataset)} train, "
              f"{len(test_loader.dataset)} test samples")

    train_loader = _subsample_loader(train_loader, max_train, rng)
    test_loader  = _subsample_loader(test_loader,  max_test,  rng)
    n_train = len(train_loader.dataset)
    n_test  = len(test_loader.dataset)
    
    if feature_space == "fc1":
        print("Collecting training fc1 features...")
        X_train_sub, y_train_sub = _collect_fc1_features(model, train_loader, device)
    elif feature_space == "pixels":
        print("Collecting training inputs...")
        X_train_sub, y_train_sub = _collect_inputs(train_loader)
    else:
        raise ValueError("feature_space must be either 'fc1' or 'pixels'")

    print("Collecting CNN predictions on test set...")
    if feature_space == "fc1":
        X_test_sub, y_test_sub, y_pred_sub = _collect_fc1_features(
            model, test_loader, device, collect_preds=True
        )
        if show_misclassifications:
            # For the misclassification image grid we need the *original pixels*.
            # In fc1 mode, X_test_sub contains 84-d features, not flattened images.
            X_test_pixels, y_test_pixels = _collect_inputs(test_loader)
            if not np.array_equal(y_test_pixels, y_test_sub):
                raise RuntimeError(
                    "Mismatch between pixel-label order and fc1-label order while "
                    "preparing misclassification image grid."
                )
    else:
        X_test_sub, y_test_sub, y_pred_sub = _collect_predictions(model, test_loader, device)
        X_test_pixels = X_test_sub

    # Combined arrays are needed for t-SNE (joint embedding) and the plot
    X_all = np.concatenate([X_train_sub, X_test_sub])
    y_all = np.concatenate([y_train_sub, y_test_sub])

    # test_mask marks the test split within the joint embedding
    test_mask = np.zeros(len(y_all), dtype=bool)
    test_mask[n_train:] = True

    n_features = X_all.shape[1]
    print(f"Running t-SNE on {len(X_all)} samples "
          f"({n_train} train + {n_test} test), {n_features} features...")
    X_2d = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=max_iter,
        init="pca",
        random_state=seed,
    ).fit_transform(X_all)

    wrong_count = int((y_test_sub != y_pred_sub).sum())
    acc = (y_test_sub == y_pred_sub).mean()
    print(f"CNN test accuracy (subsample): {acc:.1%} | "
          f"misclassified: {wrong_count}/{n_test}")

    classes_tag = ("_classes" + "-".join(str(c) for c in classes)
                   if classes is not None else "")
    # Organise outputs into clear subfolders:
    #   plots/<feature_space>/tsne/...
    #   plots/<feature_space>/misclassified/...
    tsne_out_dir = os.path.join(save_dir, feature_space, "tsne")
    mis_out_dir = os.path.join(save_dir, feature_space, "misclassified")

    save_path = os.path.join(
        tsne_out_dir,
        f"tsne_{model_name}{classes_tag}.png",
    )
    classes_str = (f"classes {classes.tolist()}"
                   if classes is not None else "all classes")
    plot_tsne_embedding_cnn(
        X_2d, y_all,
        title=(f"t-SNE of {model_name} {feature_space} features ({classes_str})\n"
               f"red \u2715 = CNN misclassification | acc: {acc:.1%}"),
        y_pred=y_pred_sub,
        test_mask=test_mask,
        save_path=save_path,
    )

    if show_misclassifications:
        if image_shape is None:
            raise ValueError("image_shape=(C, H, W) is required when show_misclassifications=True")
        errors_path = os.path.join(
            mis_out_dir,
            f"misclassified_{model_name}{classes_tag}.png",
        )
        show_misclassified_images(X_test_pixels, y_test_sub, y_pred_sub,
                                  image_shape=image_shape, classes=classes,
                                  save_path=errors_path)

    return X_2d, y_all, y_pred_sub, test_mask
