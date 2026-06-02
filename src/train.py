# =============================================================================#
#
# Datasets:
#   1. Cat–Dog (binary classification, 2 classes)
#   2. Visual Domain Decathlon Subset (multi-class, 10 classes)
#
# Experiment:
#   For each dataset, train CNNs with j = 1, 2, 3 convolutional blocks
#   across image sizes k = 32, 64, 112 → 9 configurations per dataset
# =============================================================================

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
import csv

# ── Reproducibility ──────────────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

# =============================================================================
# Configuration
# =============================================================================

def get_config():
    parser = argparse.ArgumentParser(description="CNN Image Classification Experiment")
    parser.add_argument("--catdog_train", type=str,
                        default=os.path.join("data", "cat-dog", "train"),
                        help="Path to Cat-Dog training directory")
    parser.add_argument("--catdog_test",  type=str,
                        default=os.path.join("data", "cat-dog", "test"),
                        help="Path to Cat-Dog test directory")
    parser.add_argument("--vdd_train",    type=str,
                        default=os.path.join("data", "vdd", "train"),
                        help="Path to VDD training directory")
    parser.add_argument("--vdd_test",     type=str,
                        default=os.path.join("data", "vdd", "test"),
                        help="Path to VDD test directory")
    parser.add_argument("--epochs",       type=int, default=15)
    parser.add_argument("--batch_size",   type=int, default=160)
    parser.add_argument("--output_dir",   type=str, default="outputs")
    args, _ = parser.parse_known_args()
    return args


# =============================================================================
# Data Loading
# =============================================================================

def load_dataset(directory: str, img_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all images from a directory tree into NumPy arrays.
    Subdirectory names are used as class labels (alphabetical order).

    Args:
        directory : Root folder with one sub-folder per class.
        img_size  : Images are resized to (img_size × img_size).

    Returns:
        images : float32 array of shape (N, img_size, img_size, 3), range [0, 1]
        labels : int32 array of shape (N,)
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"Dataset directory not found: '{directory}'\n"
            "Please update the --catdog_train / --vdd_train arguments or "
            "place data under data/cat-dog/ and data/vdd/ as described in README.md"
        )

    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=(img_size, img_size),
        batch_size=None,
        label_mode="int",
        shuffle=False          # keep order deterministic; we shuffle in fit()
    )

    images, labels = [], []
    for image, label in ds:
        images.append(image.numpy())
        labels.append(label.numpy())

    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels, dtype=np.int32)
    return images, labels


# =============================================================================
# Visualisation helpers
# =============================================================================

def plot_sample_grid(images: np.ndarray, labels: np.ndarray,
                     class_names: list[str], title: str, save_path: str) -> None:
    """Display a 5×5 grid of sample images with their class labels."""
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_xlabel(class_names[labels[i]], fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved sample grid → {save_path}")


def plot_training_history(history, title: str, save_path: str) -> None:
    """Plot train vs. validation accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Accuracy
    ax1.plot(history.history["accuracy"],     label="Train accuracy")
    ax1.plot(history.history["val_accuracy"], label="Val accuracy")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.set_ylim([0.4, 1.0]); ax1.legend(); ax1.grid(alpha=0.3)

    # Loss
    ax2.plot(history.history["loss"],     label="Train loss")
    ax2.plot(history.history["val_loss"], label="Val loss")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved training curves → {save_path}")


def plot_results_heatmap(results: list[dict], metric: str,
                         title: str, save_path: str) -> None:
    """
    Render a 3×3 heatmap of test accuracy (or loss) for
    all (img_size × cnn_depth) combinations.
    """
    img_sizes  = sorted(set(r["img_size"]  for r in results))
    cnn_depths = sorted(set(r["cnn_depth"] for r in results))
    matrix = np.zeros((len(img_sizes), len(cnn_depths)))

    for r in results:
        i = img_sizes.index(r["img_size"])
        j = cnn_depths.index(r["cnn_depth"])
        matrix[i, j] = r[metric]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(cnn_depths)));  ax.set_xticklabels([f"j={d}" for d in cnn_depths])
    ax.set_yticks(range(len(img_sizes)));   ax.set_yticklabels([f"{s}×{s}" for s in img_sizes])
    ax.set_xlabel("CNN Depth (conv blocks)"); ax.set_ylabel("Image Size")
    ax.set_title(title, fontweight="bold")

    for i in range(len(img_sizes)):
        for j in range(len(cnn_depths)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=11, color="black")

    plt.colorbar(im, ax=ax, label=metric.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved results heatmap → {save_path}")


# =============================================================================
# Model Builder
# =============================================================================

def build_cnn(img_size: int, num_classes: int, cnn_depth: int) -> keras.Model:
    """
    Build a Sequential CNN with `cnn_depth` convolutional blocks.

    Architecture per block:
        Conv2D(filters, 3×3, same, relu)
        BatchNormalization
        MaxPooling2D(2×2)
        Dropout(0.25)

    Classifier head:
        Flatten → Dense(128, relu) → Dropout(0.5) → Dense(num_classes)

    Args:
        img_size    : Input image side length (square images assumed).
        num_classes : Number of output classes.
        cnn_depth   : Number of conv blocks (1, 2, or 3).

    Returns:
        Compiled Keras model.
    """
    filter_sizes = [32, 64, 64]   # filters per block

    inputs = keras.Input(shape=(img_size, img_size, 3), name="input")
    x = inputs

    for block_idx in range(cnn_depth):
        filters = filter_sizes[block_idx]
        x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu",
                          name=f"conv{block_idx + 1}")(x)
        x = layers.BatchNormalization(name=f"bn{block_idx + 1}")(x)
        x = layers.MaxPooling2D((2, 2), padding="same",
                                name=f"pool{block_idx + 1}")(x)
        x = layers.Dropout(0.25, name=f"drop_conv{block_idx + 1}")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.5, name="drop_fc")(x)
    outputs = layers.Dense(num_classes, name="output")(x)

    model = keras.Model(inputs, outputs,
                        name=f"CNN_depth{cnn_depth}_img{img_size}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    return model


# =============================================================================
# Experiment Runner
# =============================================================================

def run_experiment(dataset_name: str,
                   train_dir: str,
                   test_dir: str,
                   class_names: list[str],
                   num_classes: int,
                   cfg) -> list[dict]:
    """
    Run the full grid experiment for one dataset.

    Returns a list of result dicts, one per (img_size, cnn_depth) combination.
    """
    img_sizes  = [32, 64, 112]
    cnn_depths = [1, 2, 3]
    results    = []

    out_dir = os.path.join(cfg.output_dir, dataset_name.lower().replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  DATASET: {dataset_name}  ({num_classes} classes)")
    print(f"{'='*60}")

    for img_size in img_sizes:
        print(f"\n── Image size: {img_size}×{img_size} ──")

        # Load & normalise once per image size
        print("  Loading training data...")
        train_images, train_labels = load_dataset(train_dir, img_size)
        print("  Loading test data...")
        test_images,  test_labels  = load_dataset(test_dir,  img_size)

        print(f"  Train: {train_images.shape}  |  Test: {test_images.shape}")

        # Sample grid (once per image size)
        grid_path = os.path.join(out_dir, f"samples_img{img_size}.png")
        plot_sample_grid(
            train_images, train_labels, class_names,
            title=f"{dataset_name} — Sample Images ({img_size}×{img_size})",
            save_path=grid_path
        )

        for cnn_depth in cnn_depths:
            run_label = f"{dataset_name} | img={img_size} | depth={cnn_depth}"
            print(f"\n  ── CNN depth j={cnn_depth} ──")

            model = build_cnn(img_size, num_classes, cnn_depth)
            model.summary(print_fn=lambda s: None)   # suppress verbose summary

            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_accuracy", patience=4,
                    restore_best_weights=True, verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5,
                    patience=2, verbose=1
                )
            ]

            history = model.fit(
                x=train_images, y=train_labels,
                batch_size=cfg.batch_size,
                epochs=cfg.epochs,
                validation_data=(test_images, test_labels),
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )

            test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
            print(f"  ✓ Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

            # Save training curves
            curve_path = os.path.join(out_dir, f"curves_img{img_size}_depth{cnn_depth}.png")
            plot_training_history(
                history,
                title=f"{run_label}\nTest acc = {test_acc:.4f}",
                save_path=curve_path
            )

            results.append({
                "dataset":    dataset_name,
                "img_size":   img_size,
                "cnn_depth":  cnn_depth,
                "test_acc":   round(test_acc,  4),
                "test_loss":  round(test_loss, 4),
                "epochs_run": len(history.history["accuracy"])
            })

    return results


# =============================================================================
# Summary helpers
# =============================================================================

def save_results_csv(results: list[dict], path: str) -> None:
    """Write all experiment results to a CSV file."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved → {path}")


def print_summary_table(results: list[dict]) -> None:
    """Print a formatted summary table to stdout."""
    print("\n" + "="*65)
    print(f"{'Dataset':<28} {'ImgSz':>5} {'Depth':>5} {'TestAcc':>8} {'Epochs':>6}")
    print("-"*65)
    for r in sorted(results, key=lambda x: (x["dataset"], x["img_size"], x["cnn_depth"])):
        print(f"{r['dataset']:<28} {r['img_size']:>5} {r['cnn_depth']:>5} "
              f"{r['test_acc']:>8.4f} {r['epochs_run']:>6}")
    print("="*65)


# =============================================================================
# Main
# =============================================================================

def main():
    cfg = get_config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []

    # ── Dataset 1: Cat–Dog ────────────────────────────────────────────────────
    catdog_results = run_experiment(
        dataset_name = "Cat-Dog",
        train_dir    = cfg.catdog_train,
        test_dir     = cfg.catdog_test,
        class_names  = ["cat", "dog"],
        num_classes  = 2,
        cfg          = cfg
    )
    all_results.extend(catdog_results)

    # Heatmap for Cat-Dog
    plot_results_heatmap(
        catdog_results, metric="test_acc",
        title="Cat-Dog — Test Accuracy by Image Size & CNN Depth",
        save_path=os.path.join(cfg.output_dir, "cat-dog", "heatmap_accuracy.png")
    )

    # ── Dataset 2: Visual Domain Decathlon (Subset) ───────────────────────────
    vdd_class_names = [f"{i:04d}" for i in range(1, 11)]   # '0001' … '0010'

    vdd_results = run_experiment(
        dataset_name = "VDD-Subset",
        train_dir    = cfg.vdd_train,
        test_dir     = cfg.vdd_test,
        class_names  = vdd_class_names,
        num_classes  = 10,
        cfg          = cfg
    )
    all_results.extend(vdd_results)

    # Heatmap for VDD
    plot_results_heatmap(
        vdd_results, metric="test_acc",
        title="VDD Subset — Test Accuracy by Image Size & CNN Depth",
        save_path=os.path.join(cfg.output_dir, "vdd-subset", "heatmap_accuracy.png")
    )

    # ── Combined summary ──────────────────────────────────────────────────────
    print_summary_table(all_results)

    csv_path = os.path.join(cfg.output_dir, f"results_{timestamp}.csv")
    save_results_csv(all_results, csv_path)

    print("\n✅ All experiments complete.")
    print(f"   Outputs saved to: {cfg.output_dir}/")


if __name__ == "__main__":
    main()
