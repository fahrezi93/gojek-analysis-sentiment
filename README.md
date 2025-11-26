# 📊 Sentiment Analysis - Aplikasi Ojek Online Reviews

Project skripsi untuk analisis sentiment review aplikasi ojek online (Gojek, Grab, Maxim) dari Google Play Store.

## 🎯 Tujuan Project

Melakukan **Aspect-Based Sentiment Analysis (ABSA)** terhadap review aplikasi ojek online dari Google Play Store. 

### Novelty:
Tidak hanya mengukur sentiment secara umum, tetapi **mengekstrak aspek-aspek spesifik** (driver, aplikasi, harga, layanan, keamanan) dan menganalisis sentiment untuk setiap aspek secara terpisah.

**Insight yang dihasilkan 100x lebih valuable!**

## 📱 Aplikasi yang Dianalisis

1. **Gojek** (com.gojek.app)
2. **Grab** (com.grabtaxi.passenger)
3. **Maxim** (com.taxsee.taxsee)

## 🚀 Quick Start

### Prasyarat

- Python 3.8 atau lebih tinggi
- PowerShell (Windows)
- Koneksi internet untuk scraping

### Setup Virtual Environment (Recommended)

#### Cara 1: Menggunakan Script Otomatis

```powershell
# Jalankan script setup otomatis
.\setup_venv.ps1
```

#### Cara 2: Manual Setup

```powershell
# 1. Buat virtual environment
python -m venv venv

# 2. Aktivasi virtual environment
.\venv\Scripts\Activate.ps1

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt
```

### Menjalankan Scraping

#### Opsi 1: Via Jupyter Notebook (Recommended) ⭐

```powershell
# Pastikan venv sudah aktif
jupyter notebook
```

**PENTING:** Gunakan `02_scraping_quality_reviews_absa.ipynb` untuk hasil optimal!

Kemudian buka `02_scraping_quality_reviews_absa.ipynb` dan jalankan cell secara berurutan.

#### Opsi 2: Via VS Code

1. Buka folder project di VS Code
2. Pilih Python interpreter dari venv: `.\venv\Scripts\python.exe`
3. Buka `02_scraping_quality_reviews_absa.ipynb` ⭐
4. Jalankan cell secara berurutan

**Tips:** Notebook ini sudah dioptimalkan untuk ABSA dengan:
- ✅ Filter review minimal 50 karakter
- ✅ Quality metrics tracking
- ✅ Aspect detection preview
- ✅ Smart retry mechanism

## 📁 Struktur Project

```
sentiment-analyst-ojol-review/
├── 01_scraping_playstore_reviews.ipynb        # Scraping basic (deprecated)
├── 02_scraping_quality_reviews_absa.ipynb     # Scraping OPTIMAL untuk ABSA ⭐
├── requirements.txt                            # Dependencies Python
├── setup_venv.ps1                             # Script setup otomatis
├── README.md                                  # Dokumentasi ini
├── ANALISIS_NOVELTY_ABSA.md                   # Analisis mendalam novelty ABSA
├── venv/                                      # Virtual environment (auto-generated)
└── data/                                      # Folder hasil scraping (auto-generated)
    ├── quality_reviews_absa_*.csv             # Quality reviews untuk ABSA ⭐
    ├── quality_gojek_absa_*.csv
    ├── quality_grab_absa_*.csv
    ├── quality_maxim_absa_*.csv
    ├── quality_reviews_absa_*.xlsx
    └── scraping_stats_*.json                  # Statistik scraping
```

## 📦 Dependencies

- **google-play-scraper**: Library untuk scraping Google Play Store
- **pandas**: Data manipulation dan analysis
- **numpy**: Numerical computing
- **openpyxl**: Export data ke Excel
- **jupyter**: Jupyter Notebook environment
- **ipykernel**: Python kernel untuk Jupyter

## 📊 Output Data

### Format File

Data hasil scraping akan tersimpan dalam 3 format:

1. **CSV Gabungan**: `reviews_all_apps_YYYYMMDD_HHMMSS.csv`
2. **CSV per Aplikasi**: `reviews_{app_name}_YYYYMMDD_HHMMSS.csv`
3. **Excel Multi-Sheet**: `reviews_all_apps_YYYYMMDD_HHMMSS.xlsx`

### Kolom Data

- `reviewId`: ID unik review
- `userName`: Nama user
- `review_text`: Teks review (untuk analisis sentiment)
- `rating`: Rating 1-5 bintang
- `review_date`: Tanggal review dibuat
- `thumbs_up`: Jumlah thumbs up
- `reviewCreatedVersion`: Versi app saat review dibuat
- `replyContent`: Balasan dari developer (jika ada)
- `repliedAt`: Tanggal balasan
- `app_name`: Nama aplikasi (Gojek/Grab/Maxim)
- `app_id`: Package ID aplikasi

## 🎯 Target Data

- **Per Aplikasi**: Minimal 3.000 review
- **Total**: ~9.000 review dari 3 aplikasi

## ⚙️ Konfigurasi Scraping

Edit pada cell ke-6 di notebook untuk mengubah konfigurasi:

```python
# Target jumlah review per aplikasi
TARGET_REVIEWS = 3000  # Ubah sesuai kebutuhan

# Bahasa dan negara
lang='id'      # Indonesia
country='id'   # Indonesia
```

## 🔍 Fitur Notebook

1. ✅ Scraping otomatis dengan pagination
2. ✅ Progress tracking real-time
3. ✅ Error handling
4. ✅ Rate limiting prevention
5. ✅ Statistik dan analisis deskriptif
6. ✅ Preview sample data
7. ✅ Export multiple format
8. ✅ Timestamp otomatis untuk file

## 🛠️ Troubleshooting

### Error: Execution Policy

Jika mendapat error saat menjalankan script PowerShell:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Error: Module Not Found

```powershell
# Pastikan venv aktif
.\venv\Scripts\Activate.ps1

# Install ulang requirements
pip install -r requirements.txt
```

### Error: Connection/Rate Limiting

- Tambah delay pada kode scraping
- Kurangi jumlah request per batch
- Gunakan VPN jika IP di-block

## 📝 Tips Penggunaan

1. **Jalankan scraping di luar jam sibuk** untuk menghindari rate limiting
2. **Backup data secara berkala** selama proses scraping
3. **Monitor progress** melalui output cell di notebook
4. **Validasi data** setelah scraping selesai
5. **Dokumentasikan parameter** scraping yang digunakan

## 🔜 Next Steps - ABSA Pipeline

Setelah data berkualitas berhasil di-scrape, tahap selanjutnya:

### 1. **Data Preprocessing** (Notebook 03)
   - Text cleaning (keep meaningful words)
   - Tokenization
   - POS Tagging
   - Stopword removal (careful - keep aspect-related words!)

### 2. **Aspect Extraction** (Notebook 04)
   - Rule-based: Keyword matching
   - Statistical: LDA Topic Modeling
   - ML-based: NER (Named Entity Recognition)
   - Deep Learning: BiLSTM-CRF

### 3. **Sentiment Classification per Aspect** (Notebook 05)
   - Traditional ML: SVM, Naive Bayes
   - Deep Learning: LSTM, BiLSTM
   - Transformer: BERT, IndoBERT
   - Attention mechanism for aspect-sentiment pairing

### 4. **Evaluation** (Notebook 06)
   - Aspect Extraction: Precision, Recall, F1
   - Sentiment per Aspect: Accuracy, F1-Score
   - End-to-End ABSA: Exact Match, Soft Match
   - Visualization & Business Insights

### 5. **Deployment & Insights**
   - Interactive dashboard
   - Actionable recommendations per aplikasi
   - Comparative analysis: Gojek vs Grab vs Maxim

## 📧 Kontributor

Project Skripsi - Sentiment Analysis Ojek Online Reviews

## 📄 License

Educational Purpose - Project Skripsi

---

**Note**: Project ini dibuat untuk keperluan penelitian/skripsi. Gunakan data dengan bijak dan patuhi Terms of Service Google Play Store.
