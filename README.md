# Diagnosis Dini Penyakit Daun Padi

Proyek tugas besar Deep Learning untuk mengklasifikasikan penyakit daun padi dari foto menggunakan **transfer learning EfficientNetB0**, membandingkannya dengan CNN baseline, dan menyediakan prototipe Streamlit yang menampilkan prediksi serta tingkat keyakinan.

## Anggota dan jobdesk

| Anggota | Peran | Tanggung jawab |
|---|---|---|
| **Randy** | Data Engineer | Mengunduh dan memeriksa dataset; EDA dan distribusi kelas; menyusun split train/validasi/test tanpa kebocoran; preprocessing dan augmentasi; mendokumentasikan sumber serta lisensi dataset. |
| **Zildan** | Model Engineer 1 | Membangun CNN baseline dari nol; melatih dan menyimpan model; mencatat konfigurasi eksperimen dan artefak untuk evaluasi. |
| **Christoper** | Model Engineer 2 | Menyiapkan EfficientNetB0 pretrained; melatih classifier head dan melakukan fine-tuning; menyimpan model terbaik serta mendokumentasikan konfigurasi dan urutan label kelas. |
| **Afzal** | Evaluator | Menetapkan protokol evaluasi yang sama untuk kedua model; menjalankan minimal 3 seed; merangkum Accuracy, Precision, Recall, dan F1 (rerata dan simpangan baku); membuat confusion matrix, analisis galat, studi ablasi, dan Grad-CAM. |
| **Samuel** | System Engineer | Menyiapkan struktur repo dan dokumentasi; membangun aplikasi Streamlit; mengintegrasikan model dan hasil prediksi; menyiapkan instruksi menjalankan aplikasi dan video demo. |

Setiap anggota menulis bagian laporan yang sesuai dengan pekerjaannya. Semua anggota ikut meninjau hasil dan membantu memastikan notebook dapat dijalankan ulang.

## Dataset

- Sumber: [Rice Disease Dataset di Kaggle](https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset)
- Rencana awal menggunakan 3.829 gambar dan 6 kelas; verifikasi jumlah gambar dan nama kelas setelah dataset diunduh.
- Rencana split: 70% train, 15% validasi, dan 15% test. Gunakan split yang sama untuk kedua model. Data test tidak digunakan untuk memilih model atau hyperparameter.
- Simpan data lokal di `data/raw/`. Dataset tidak disertakan di GitHub; ikuti ketentuan lisensi dan syarat penggunaan Kaggle. Lihat [data/README.md](data/README.md).

## Metode yang direncanakan

- Resize gambar ke 224×224 dan lakukan preprocessing yang sesuai dengan model.
- Augmentasi data train: flip, rotasi, zoom, dan brightness. Data validasi dan test tidak diaugmentasi.
- **Baseline:** CNN sederhana yang dilatih dari awal.
- **Model usulan:** EfficientNetB0 pretrained dengan fine-tuning.
- **Evaluasi:** minimal 3 seed; Accuracy, Precision, Recall, F1; confusion matrix; studi ablasi; dan Grad-CAM.
- **Prototipe:** Streamlit untuk mengunggah gambar, melihat prediksi dan confidence, serta membaca informasi penanganan yang telah dirujuk.

> Nama kelas, parameter eksperimen, hasil metrik, dan informasi penanganan akan diisi setelah diverifikasi. Prediksi prototipe adalah alat bantu awal, bukan pengganti diagnosis penyuluh.

## Struktur repository

```text
.
├── app/
│   └── app.py
├── data/
│   ├── README.md
│   ├── raw/                 # data Kaggle lokal; tidak di-commit
│   └── processed/           # hasil preprocessing lokal; tidak di-commit
├── docs/                    # catatan desain dan dokumentasi
├── models/                  # model terlatih lokal; tidak di-commit
├── notebooks/
│   ├── 01_eda_and_split.ipynb
│   ├── 02_cnn_baseline.ipynb
│   ├── 03_efficientnet.ipynb
│   └── 04_evaluation_and_gradcam.ipynb
├── reports/
│   └── figures/             # grafik dan visualisasi laporan
├── src/
│   ├── data/
│   ├── models/
│   ├── evaluation/
│   └── inference/
├── .gitignore
├── README.md
└── requirements.txt
```

File notebook dan modul pada struktur di atas adalah rancangan lokasi kerja; tambahkan implementasinya saat tersedia.

## Menjalankan aplikasi

Setelah dependensi, aplikasi, dan berkas model disiapkan:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app/app.py
```

## Reproduksibilitas dan kolaborasi

- Catat seed, versi library, hyperparameter, dan hasil setiap run.
- Pastikan urutan label kelas konsisten antara pelatihan, evaluasi, dan aplikasi.
- Jangan gunakan data test untuk memilih konfigurasi.
- Kerjakan perubahan pada branch masing-masing dan ajukan Pull Request sebelum menggabungkannya ke `main`.
- Jangan commit dataset, checkpoint besar, file rahasia, atau environment lokal.

## Referensi

- [Anshul M. — Rice Disease Dataset, Kaggle](https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset)
- Tugas besar dan format laporan mengikuti dokumen instruksi mata kuliah Deep Learning.
