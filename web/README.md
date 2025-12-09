# 🚗 Aplikasi Web Analisis Sentimen Gojek - IndoBERT

Aplikasi web berbasis Streamlit untuk menganalisis sentimen ulasan pelanggan Gojek menggunakan model IndoBERT.

## 📋 Deskripsi

Aplikasi ini terdiri dari 4 modul utama sesuai dengan arsitektur OOP yang terstruktur:

### 🔧 Struktur Kelas

| Nama Kelas | File | Deskripsi |
|------------|------|-----------|
| **SentimentUI** | `app.py` | Kelas utama yang mengatur tata letak dan logika antarmuka Streamlit. Menangani interaksi pengguna, pemilihan skema klasifikasi (3 atau 5 kelas), unggah file CSV, dan visualisasi hasil prediksi. |
| **ModelPredictor** | `predictor.py` | Bertanggung jawab atas logika inferensi. Memuat bobot model IndoBERT yang telah dilatih, menerima input token, dan menghasilkan output berupa label sentimen beserta skor probabilitasnya. |
| **TextNormalizer** | `preprocessing.py` | Menangani seluruh tahapan pembersihan data mentah. Fungsi utama meliputi penghapusan noise (emoji, simbol), normalisasi kata slang khas ulasan Gojek, dan konversi teks agar siap ditokenisasi. |
| **PerformanceEvaluator** | `evaluation.py` | Kelas khusus untuk halaman evaluasi yang bertugas menghitung metrik statistik (Akurasi, Presisi, Recall, F1-Score) dan menyusun data untuk visualisasi Confusion Matrix berdasarkan data uji yang diunggah. |

## ✨ Fitur Utama

### 🏠 Halaman Beranda
- Dashboard dengan overview performa model
- Statistik metrik utama (Akurasi, Presisi, Recall, F1-Score)
- Panduan penggunaan aplikasi

### 🔮 Analisis Sentimen
- **Input Manual**: Analisis sentimen untuk teks yang diketik langsung
- **Upload CSV**: Analisis batch untuk multiple ulasan
- **Real-time Prediction**: Hasil prediksi instan dengan confidence score
- **Visualisasi Probabilitas**: Distribusi probabilitas untuk semua kelas
- **Detail Preprocessing**: Lihat perbandingan teks sebelum dan sesudah preprocessing
- **Export Hasil**: Download hasil analisis dalam format CSV

### 📊 Evaluasi Model
- Upload data uji (test set) dengan ground truth labels
- Metrik evaluasi komprehensif:
  - Akurasi, Presisi, Recall, F1-Score
  - Metrik per-kelas
  - Confusion Matrix (visualisasi interaktif)
  - Distribusi prediksi
  - Classification Report lengkap
- Visualisasi interaktif dengan Plotly

### ℹ️ Tentang
- Informasi proyek dan teknologi yang digunakan
- Arsitektur model IndoBERT
- Detail dataset dan preprocessing
- Dokumentasi fitur aplikasi

## 🛠️ Teknologi

### Backend
- **Model**: IndoBERT Base (indobenchmark/indobert-base-p1)
- **Framework**: PyTorch 2.0.1
- **Transformers**: HuggingFace Transformers 4.33.0

### Frontend
- **Web Framework**: Streamlit 1.28.0
- **Visualisasi**: Plotly 5.17.0
- **Data Processing**: Pandas, NumPy

### Text Processing
- Custom TextNormalizer dengan dictionary slang Indonesia
- Regex-based cleaning
- Tokenisasi dengan IndoBERT tokenizer

## 🚀 Instalasi

### 1. Install Dependencies

```bash
cd web
pip install -r requirements.txt
```

### 2. Struktur Model

Pastikan Anda memiliki model yang sudah dilatih di lokasi berikut:

```
sentiment-analyst-ojol-review/
├── saved_model_indobert_3class/     # Model 3-kelas
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── vocab.txt
└── saved_model_indobert_5class/     # Model 5-kelas (opsional)
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    └── vocab.txt
```

### 3. Jalankan Aplikasi

```bash
streamlit run app.py
```

Atau gunakan PowerShell script:

```bash
..\run_web_app.ps1
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## 📖 Cara Menggunakan

### Analisis Sentimen Manual

1. Pilih **Skema Model** di sidebar (3-Kelas atau 5-Kelas)
2. Klik **🔄 Load Model** untuk memuat model
3. Navigasi ke **🔮 Analisis Sentimen**
4. Pilih **💬 Teks Manual**
5. Ketik atau paste ulasan
6. Klik **🔍 ANALISIS SENTIMEN**
7. Lihat hasil prediksi dengan confidence score

### Analisis Batch dari CSV

1. Navigasi ke **🔮 Analisis Sentimen**
2. Pilih **📄 Upload File CSV**
3. Upload file CSV dengan kolom `text` atau `review`
4. Klik **🔍 ANALISIS SEMUA DATA**
5. Lihat ringkasan hasil dan distribusi sentimen
6. Download hasil dengan **⬇️ Download Hasil (CSV)**

Format CSV:
```csv
text
"Aplikasi sangat membantu, driver ramah"
"Pelayanan buruk, lama banget"
"Biasa saja, tidak ada yang spesial"
```

### Evaluasi Model

1. Navigasi ke **📊 Evaluasi Model**
2. Upload file CSV dengan kolom `text` dan `label`
3. Klik **🎯 MULAI EVALUASI**
4. Lihat metrik evaluasi:
   - Metrik utama (Akurasi, Presisi, Recall, F1-Score)
   - Confusion Matrix
   - Metrik per-kelas
   - Distribusi prediksi
   - Classification Report

Format CSV untuk evaluasi:
```csv
text,label
"Aplikasi sangat membantu, driver ramah",2
"Pelayanan buruk, lama banget",0
"Biasa saja, tidak ada yang spesial",1
```

Label untuk 3-kelas:
- 0 = Negatif
- 1 = Netral
- 2 = Positif

Label untuk 5-kelas:
- 0 = Rating 1 (Sangat Negatif)
- 1 = Rating 2 (Negatif)
- 2 = Rating 3 (Netral)
- 3 = Rating 4 (Positif)
- 4 = Rating 5 (Sangat Positif)

## 🎨 Fitur UI

### Design Highlights
- ✅ Modern gradient design dengan warna Gojek (#00AA13)
- ✅ Responsive layout dengan Streamlit columns
- ✅ Interactive visualizations dengan Plotly
- ✅ Custom CSS styling untuk tampilan professional
- ✅ Emoji dan icons untuk UX yang lebih baik
- ✅ Progress indicators untuk operasi yang memakan waktu
- ✅ Collapsible sections untuk detail tambahan

### Color Scheme
- **Gojek Green**: #00AA13 (Primary color)
- **Success**: #10B981 (Positive sentiment)
- **Warning**: #FFC107 (Neutral sentiment)
- **Danger**: #DC3545 (Negative sentiment)
- **Info**: #2196F3 (Informational elements)

## 📊 Contoh Output

### Prediksi Single Text
```
Sentimen: POSITIF 😊
Confidence: 98.5%

Probabilitas:
- Positif: 98.5%
- Netral: 1.2%
- Negatif: 0.3%
```

### Evaluasi Model
```
Metrik Utama:
- Akurasi: 92%
- Presisi: 91%
- Recall: 90%
- F1-Score: 91%

Confusion Matrix: [Visualisasi Heatmap]
Metrik Per Kelas: [Grouped Bar Chart]
Distribusi Prediksi: [Pie Chart]
```

## 🔧 Troubleshooting

### Model tidak ditemukan
Pastikan Anda sudah melatih model dengan menjalankan notebook training terlebih dahulu:
- `Training_IndoBERT_3Class.ipynb` untuk model 3-kelas
- `Training_IndoBERT_5Class.ipynb` untuk model 5-kelas

### Error saat load model
Periksa:
- Path model sudah benar
- File model lengkap (config.json, model.safetensors, vocab.txt)
- Dependencies sudah terinstall

### CSV upload error
Pastikan:
- File format CSV dengan encoding UTF-8
- Kolom `text` atau `review` ada untuk analisis
- Kolom `label` ada untuk evaluasi
- Tidak ada missing values di kolom penting

## 📝 Lisensi

© 2024 - Sentiment Analysis for Gojek Reviews

## 👨‍💻 Developer

Aplikasi ini dibuat sebagai bagian dari skripsi untuk menganalisis sentimen ulasan Gojek menggunakan model IndoBERT.

---

**Built with ❤️ using Streamlit & IndoBERT**
