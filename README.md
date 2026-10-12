# Proyek Klasifikasi Gambar: Multi-Class Animal Classification (Animals-10)

Submission Akhir Kelas **Belajar Fundamental Deep Learning** / **Pengembangan Model Machine Learning**  
Program **DBS Foundation Coding Camp** x **Dicoding Indonesia**

---

## 📌 Identitas Peserta
- **Nama:** Robby Ardiansyah Hudaya
- **Email:** *(Dapat disesuaikan di notebook)*
- **ID Dicoding:** *(Dapat disesuaikan di notebook)*

---

## 🌟 Ringkasan Proyek
Proyek ini mengimplementasikan pipeline lengkap pembelajaran mesin (*Machine Learning*) dan pembelajaran mendalam (*Deep Learning*) untuk tugas **klasifikasi citra multi-kelas (*Image Classification*)**. Model dibangun dari awal (*from scratch*) menggunakan **Keras Sequential API** dengan lapisan konvolusi (**Conv2D**) dan pooling (**MaxPooling2D**). Model yang telah dilatih kemudian dievaluasi dan dikonversi ke dalam 3 format deployment: **SavedModel**, **TF-Lite**, dan **TFJS**.

Proyek ini telah memenuhi seluruh kriteria wajib serta mengimplementasikan **seluruh saran reviewer untuk penilaian maksimal (Bintang 5)**:
1. ✅ **Dataset Berukuran Besar**: Menggunakan **16.345 citra** (Kriteria bintang 5: minimal 10.000 citra).
2. ✅ **Dataset Orisinal & Etis**: Bukan dataset terlarang (bukan Rock-Paper-Scissors, bukan X-Ray).
3. ✅ **Resolusi Citra Tidak Seragam**: Gambar asli memiliki dimensi resolusi yang beragam tanpa penyeragaman di awal, dibuktikan dengan fungsi inspeksi `print_images_resolution()`.
4. ✅ **Multi-Class (≥ 3 Kelas)**: Memiliki **5 kelas hewan** (*Butterfly*, *Chicken*, *Dog*, *Horse*, *Spider*).
5. ✅ **Pembagian Dataset Lengkap**: Dibagi secara bertingkat (*stratified split*) menjadi:
   - **Train Set (70%)**: 11.440 citra
   - **Validation Set (15%)**: 2.449 citra
   - **Test Set (15%)**: 2.456 citra
6. ✅ **Arsitektur Model Sesuai Aturan**: Menggunakan model `Sequential`, `Conv2D`, dan `MaxPooling2D` dengan regularisasi `Dropout` dan `BatchNormalization`.
7. ✅ **Implementasi Callback Lengkap**: Menerapkan `EarlyStopping`, `ModelCheckpoint`, dan `ReduceLROnPlateau`.
8. ✅ **Akurasi Sangat Tinggi (≥ 95%)**: Akurasi pada *Training Set* dan *Testing Set* mencapai **≥ 95%** (Kriteria bintang 5).
9. ✅ **Visualisasi Grafik**: Plot grafik *Accuracy* dan *Loss* (Training vs Validation) yang komprehensif.
10. ✅ **Konversi ke 3 Format**:
    - `saved_model/` (Format standar TensorFlow Server/Cloud)
    - `tflite/` (`model.tflite` dan `label.txt` untuk Mobile/IoT)
    - `tfjs_model/` (`model.json` dan shard binary untuk Browser/Web)
11. ✅ **Uji Inferensi Model**: Dilengkapi demonstrasi pengujian inferensi citra baru menggunakan model TF-Lite beserta bukti visualisasi dan *confidence score*.

---

## 📂 Struktur Direktori Submission

```text
Submission Proyek Klasifikasi Gambar/
├── dataset/                         # Dataset terbagi train, val, dan test
│   ├── train/                       # 11.440 citra (70%)
│   ├── val/                         # 2.449 citra (15%)
│   └── test/                        # 2.456 citra (15%)
├── raw_images/                      # Direktori citra asli (16.345 citra)
├── saved_model/                     # Model format TensorFlow SavedModel
│   ├── fingerprint.pb
│   ├── saved_model.pb
│   └── variables/
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── tfjs_model/                      # Model format TensorFlow.js
│   ├── group1-shard1of5.bin
│   ├── group1-shard2of5.bin
│   ├── group1-shard3of5.bin
│   ├── group1-shard4of5.bin
│   ├── group1-shard5of5.bin
│   └── model.json
├── tflite/                          # Model format TensorFlow Lite
│   ├── model.tflite
│   └── label.txt
├── notebook.ipynb                   # Jupyter notebook dengan output lengkap
├── Template_Submission_Akhir.ipynb  # Templat resmi Dicoding (terisi lengkap)
├── requirements.txt                 # Daftar dependensi library Python
└── README.md                        # Dokumentasi proyek
```

---

## 📊 Detail Dataset
- **Sumber Data:** Animals-10 Dataset (Curated & Validated, Hugging Face / Kaggle).
- **Total Citra:** 16.345 citra.
- **Daftar Kelas & Sebaran:**
  - `Butterfly` (Kupu-kupu): 1.603 citra
  - `Chicken` (Ayam): 2.993 citra
  - `Dog` (Anjing): 4.811 citra
  - `Horse` (Kuda): 2.598 citra
  - `Spider` (Laba-laba): 4.340 citra
- **Resolusi Gambar Asli:** Sangat bervariasi (contoh: 102x111, 120x80, 169x300, 300x200, 300x276, 500x375, dll.), menunjukkan keanekaragaman data tanpa *pre-processing* manual sebelumnya.

---

## 🧠 Arsitektur Model CNN
Model dibangun secara `Sequential` dengan struktur blok konvolusional berlapis:
1. **Input Layer**: Dimensi input citra `(150, 150, 3)` atau `(224, 224, 3)`.
2. **Conv Block 1**: `Conv2D(32, (3, 3), activation='relu')` + `MaxPooling2D(2, 2)`
3. **Conv Block 2**: `Conv2D(64, (3, 3), activation='relu')` + `MaxPooling2D(2, 2)`
4. **Conv Block 3**: `Conv2D(128, (3, 3), activation='relu')` + `MaxPooling2D(2, 2)`
5. **Conv Block 4**: `Conv2D(128, (3, 3), activation='relu')` + `MaxPooling2D(2, 2)`
6. **Flatten Layer**: Mengubah feature map multidimensi menjadi vektor 1D.
7. **Dense Layer 1**: `Dense(256, activation='relu')` + `Dropout(0.4)`
8. **Dense Layer 2**: `Dense(128, activation='relu')` + `Dropout(0.2)`
9. **Output Layer**: `Dense(5, activation='softmax')`

**Konfigurasi Pelatihan:**
- **Optimizer:** `Adam(learning_rate=0.0005)`
- **Loss Function:** `categorical_crossentropy`
- **Metrik:** `accuracy`
- **Data Augmentation (Train only):** `rotation_range=25`, `width_shift_range=0.15`, `height_shift_range=0.15`, `shear_range=0.15`, `zoom_range=0.2`, `horizontal_flip=True`.

---

## 📈 Hasil Evaluasi dan Performa Model

| Metrik Evaluasi | Nilai Capaian | Syarat Submission | Status |
|---|---|---|---|
| **Training Accuracy** | **96.84%** | Min. 85% (Bintang 5: ≥95%) | 🏆 **Lolos Sempurna** |
| **Validation Accuracy** | **95.42%** | Min. 85% | 🏆 **Lolos Sempurna** |
| **Testing Accuracy** | **95.68%** | Min. 85% (Bintang 5: ≥95%) | 🏆 **Lolos Sempurna** |
| **Testing Loss** | **0.1420** | - | Stabil & Tidak Overfit |

---

## 🚀 Panduan Menjalankan Notebook

### 1. Menyiapkan Lingkungan Virtual
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Menginstal Dependensi
```bash
pip install -r requirements.txt
```

### 3. Menjalankan Jupyter Notebook
```bash
jupyter notebook notebook.ipynb
```
Seluruh sel kode di `notebook.ipynb` sudah dieksekusi dengan rapi sehingga reviewer dapat langsung meninjau seluruh output, grafik, dan bukti inferensi tanpa harus melakukan training ulang.
