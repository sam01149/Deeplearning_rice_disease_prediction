"""Utilitas data bersama untuk proyek Diagnosis Dini Penyakit Daun Padi.

Dipakai oleh semua notebook (baseline, EfficientNet, evaluasi) dan aplikasi Streamlit
agar pembagian data, urutan label, dan pra-pemrosesan selalu sama.

Contoh di notebook model:
    import sys; sys.path.append("..")            # dari folder notebooks/
    from src.data.data_utils import make_dataset, CLASS_NAMES, NUM_CLASSES
    train_ds = make_dataset("train")             # sudah teraugmentasi
    val_ds   = make_dataset("val")
    test_ds  = make_dataset("test")

Normalisasi dilakukan di dalam model:
    - CNN baseline : tambahkan layers.Rescaling(1/255) di awal model
    - EfficientNet : input dibiarkan 0-255 (normalisasi bawaan Keras)
"""
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- konfigurasi bersama
SEED = 42
IMG_SIZE = 224                  # ukuran input model
STORE_SHORT_SIDE = 256          # sisi terpendek gambar yang disimpan di data/processed
REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "rice_256"
SPLITS_CSV = REPO_ROOT / "data" / "splits.csv"


def _class_names():
    if SPLITS_CSV.exists():
        return sorted(pd.read_csv(SPLITS_CSV)["label"].unique().tolist())
    return []


CLASS_NAMES = _class_names()    # urutan label = urutan abjad nama folder; WAJIB sama di model dan aplikasi
NUM_CLASSES = len(CLASS_NAMES)
CLASS_INDEX = {c: i for i, c in enumerate(CLASS_NAMES)}


# ---------------------------------------------------------------- pra-pemrosesan gambar (PIL)
def shrink_pil(img, short_side=STORE_SHORT_SIDE):
    """Putar sesuai EXIF, ubah ke RGB, kecilkan sisi terpendek ke `short_side` (rasio dipertahankan).

    Dipakai saat membuat data/processed DAN di aplikasi, supaya gambar training dan gambar
    pengguna diproses dengan cara yang sama.
    """
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    scale = short_side / min(w, h)
    if scale < 1:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return img


def load_for_inference(pil_image):
    """Gambar dari pengguna (PIL) -> array float32 (1, IMG_SIZE, IMG_SIZE, 3) rentang 0-255, siap model.predict."""
    import tensorflow as tf
    arr = np.asarray(shrink_pil(pil_image), dtype=np.float32)
    arr = tf.image.resize(arr, (IMG_SIZE, IMG_SIZE)).numpy()
    return arr[None, ...]


# ---------------------------------------------------------------- tf.data pipeline
def build_augmentation(seed=SEED):
    """Augmentasi khusus data train: meniru variasi foto petani di lapangan (tanpa mengubah warna/hue)."""
    from tensorflow import keras
    from tensorflow.keras import layers
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=seed),
        layers.RandomRotation(15 / 360, seed=seed),
        layers.RandomZoom(0.10, seed=seed),
        layers.RandomBrightness(0.20, value_range=(0, 255), seed=seed),
        layers.RandomContrast(0.20, seed=seed),
    ], name="augmentasi")


def _load_image(path, label):
    import tensorflow as tf
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))          # float32, rentang 0-255
    return img, label


def make_dataset(split, batch_size=32, augment=None, seed=SEED, splits_csv=SPLITS_CSV, image_dir=PROCESSED_DIR):
    """Dataset tf.data untuk 'train', 'val', atau 'test'.

    Label berupa indeks integer (pakai loss sparse_categorical_crossentropy).
    Augmentasi default hanya aktif untuk 'train'. Untuk studi ablasi tanpa augmentasi: augment=False.
    """
    import tensorflow as tf
    if split not in ("train", "val", "test"):
        raise ValueError("split harus 'train', 'val', atau 'test'")
    d = pd.read_csv(splits_csv)
    names = sorted(d["label"].unique().tolist())
    index = {c: i for i, c in enumerate(names)}
    d = d[d["split"] == split]
    paths = [str(Path(image_dir) / p) for p in d["relpath"]]
    labels = d["label"].map(index).astype("int32").values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if split == "train":
        ds = ds.shuffle(len(d), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    if augment is None:
        augment = split == "train"
    if augment:
        aug = build_augmentation(seed)
        ds = ds.map(lambda x, y: (tf.clip_by_value(aug(x, training=True), 0, 255), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def split_labels(split, splits_csv=SPLITS_CSV):
    """Label integer (urutan sama dengan make_dataset tanpa shuffle) untuk 'val'/'test', dipakai evaluator."""
    d = pd.read_csv(splits_csv)
    index = {c: i for i, c in enumerate(sorted(d["label"].unique()))}
    return d[d["split"] == split]["label"].map(index).values
