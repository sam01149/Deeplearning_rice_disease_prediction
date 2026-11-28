"""CNN baseline (dilatih dari nol) untuk klasifikasi penyakit daun padi.

Model Engineer 1. Dipakai oleh notebooks/02_cnn_baseline.ipynb dan oleh evaluator
untuk menjalankan ulang pelatihan dengan seed berbeda.

Contoh:
    import sys; sys.path.append("..")
    from src.models.cnn_baseline import train_cnn
    model, history, cfg = train_cnn(seed=42)
"""
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import data_utils as du

MODELS_DIR = du.REPO_ROOT / "models"
REPORTS_DIR = du.REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Hyperparameter default (catat perubahan di laporan bila diubah)
DEFAULT_CONFIG = {
    "model": "cnn_baseline",
    "img_size": du.IMG_SIZE,
    "filters": [32, 64, 128, 256],
    "dense_units": 128,
    "dropout": 0.5,
    "bn_momentum": 0.9,
    "l2": 1e-4,
    "optimizer": "adam",
    "learning_rate": 3e-4,
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 20,
    "reduce_lr_patience": 3,
    "reduce_lr_factor": 0.5,
    "monitor": "val_loss",
    "augment": True,
    "class_weight": True,
}


def set_seed(seed):
    """Seed Python, NumPy, dan TensorFlow sekaligus."""
    from tensorflow import keras
    keras.utils.set_random_seed(seed)


def build_cnn(num_classes=du.NUM_CLASSES, img_size=du.IMG_SIZE, filters=(32, 64, 128, 256),
              dense_units=128, dropout=0.5, l2=1e-4, bn_momentum=0.9, name="cnn_baseline"):
    """CNN sederhana: 4 blok [Conv-BN-ReLU x2 -> MaxPool] -> GlobalAveragePooling -> Dense -> Softmax.

    Input rentang 0-255 (dari make_dataset), dinormalisasi di dalam model dengan Rescaling(1/255).
    GlobalAveragePooling dipakai (bukan Flatten) agar jumlah parameter kecil, karena data latih
    hanya ~1.400 gambar dan model dari nol mudah overfit.
    Momentum BatchNorm 0.9 (default Keras 0.99) agar statistik inference cepat stabil;
    dengan 0.99 akurasi validasi sempat tertahan di ~20% pada percobaan awal.
    """
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    reg = regularizers.l2(l2) if l2 else None
    inputs = keras.Input(shape=(img_size, img_size, 3), name="gambar")
    x = layers.Rescaling(1.0 / 255, name="normalisasi")(inputs)
    for i, f in enumerate(filters, start=1):
        for j in (1, 2):
            x = layers.Conv2D(f, 3, padding="same", use_bias=False, kernel_regularizer=reg,
                              name=f"blok{i}_conv{j}")(x)
            x = layers.BatchNormalization(momentum=bn_momentum, name=f"blok{i}_bn{j}")(x)
            x = layers.ReLU(name=f"blok{i}_relu{j}")(x)
        x = layers.MaxPooling2D(2, name=f"blok{i}_pool")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout1")(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=reg, name="fc")(x)
    x = layers.Dropout(dropout, name="dropout2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="prediksi")(x)
    return keras.Model(inputs, outputs, name=name)


def compute_class_weights(splits_csv=du.SPLITS_CSV):
    """Bobot kelas 'balanced' dari data train: total / (jumlah_kelas * jumlah_per_kelas)."""
    labels = du.split_labels("train", splits_csv)
    counts = np.bincount(labels, minlength=du.NUM_CLASSES)
    return {i: float(len(labels) / (du.NUM_CLASSES * c)) for i, c in enumerate(counts)}


def _versions():
    import tensorflow as tf
    return {"python": platform.python_version(), "tensorflow": tf.__version__,
            "numpy": np.__version__, "pandas": pd.__version__}


def train_cnn(seed=du.SEED, tag=None, verbose=1, **overrides):
    """Latih CNN baseline untuk satu seed dan simpan semua artefak.

    Artefak (tag default: f"cnn_baseline_seed{seed}"):
        models/<tag>.keras                 model terbaik (berdasarkan val_loss)
        reports/<tag>_history.csv          loss/accuracy per epoch
        reports/<tag>_config.json          seed, hyperparameter, versi library, urutan kelas, hasil val
    Data test TIDAK dipakai di sini.
    """
    from tensorflow import keras

    cfg = {**DEFAULT_CONFIG, **overrides, "seed": seed}
    tag = tag or f"cnn_baseline_seed{seed}"
    for d in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    train_ds = du.make_dataset("train", batch_size=cfg["batch_size"], augment=cfg["augment"], seed=seed)
    val_ds = du.make_dataset("val", batch_size=cfg["batch_size"])

    model = build_cnn(filters=cfg["filters"], dense_units=cfg["dense_units"],
                      dropout=cfg["dropout"], l2=cfg["l2"], bn_momentum=cfg["bn_momentum"])
    model.compile(optimizer=keras.optimizers.Adam(cfg["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model_path = MODELS_DIR / f"{tag}.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(model_path), monitor=cfg["monitor"], save_best_only=True),
        keras.callbacks.EarlyStopping(monitor=cfg["monitor"], patience=cfg["early_stopping_patience"],
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor=cfg["monitor"], factor=cfg["reduce_lr_factor"],
                                          patience=cfg["reduce_lr_patience"], min_lr=1e-6),
    ]
    class_weight = compute_class_weights() if cfg["class_weight"] else None

    history = model.fit(train_ds, validation_data=val_ds, epochs=cfg["epochs"],
                        class_weight=class_weight, callbacks=callbacks, verbose=verbose)

    hist = pd.DataFrame(history.history)
    hist.index.name = "epoch"
    hist.index += 1
    hist.to_csv(REPORTS_DIR / f"{tag}_history.csv")

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    best_epoch = int(hist["val_loss"].idxmin())
    cfg.update({
        "tag": tag,
        "class_names": du.CLASS_NAMES,
        "class_weight_values": class_weight,
        "n_params": int(model.count_params()),
        "epochs_run": int(len(hist)),
        "best_epoch": best_epoch,
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "model_path": str(model_path.relative_to(du.REPO_ROOT)),
        "versions": _versions(),
    })
    with open(REPORTS_DIR / f"{tag}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return model, history, cfg


def plot_history(history_csv, out_png=None, title="CNN baseline"):
    """Plot kurva loss dan accuracy train vs val dari file history CSV."""
    import matplotlib.pyplot as plt
    h = pd.read_csv(history_csv, index_col="epoch")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, m in zip(axes, ("loss", "accuracy")):
        ax.plot(h.index, h[m], label="train")
        ax.plot(h.index, h[f"val_{m}"], label="validasi")
        ax.axvline(h["val_loss"].idxmin(), ls="--", c="gray", lw=1, label="epoch terbaik")
        ax.set_xlabel("epoch"); ax.set_ylabel(m); ax.set_title(f"{title} - {m}"); ax.legend()
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
    return fig
