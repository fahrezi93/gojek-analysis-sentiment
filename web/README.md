# 🛵 Sentiment Analysis Web App

Aplikasi web untuk menganalisis sentimen review aplikasi ojek online menggunakan model IndoBERT.

## 📋 Fitur

- ✅ Analisis sentimen real-time
- ✅ 3 kelas sentimen: Positif, Netral, Negatif
- ✅ Visualisasi probabilitas per kelas
- ✅ Contoh review yang bisa dicoba langsung
- ✅ UI modern dan responsive

## 🚀 Cara Menjalankan

### Prasyarat

1. Pastikan sudah menginstall semua dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Pastikan model sudah dilatih dengan menjalankan notebook `sentiment_training_3class_final.ipynb`. 
   Model akan disimpan di folder `models/`.

### Menjalankan Aplikasi

Dari root folder project:

```bash
streamlit run web/app.py
```

Atau menggunakan script PowerShell:

```powershell
.\run_web_app.ps1
```

Aplikasi akan terbuka di browser pada alamat: `http://localhost:8501`

## 📁 Struktur File

```
web/
├── app.py          # Main Streamlit application
└── README.md       # Dokumentasi ini

models/
├── indobert_sentiment_3class.pt  # Trained model weights
├── tokenizer/                     # BERT tokenizer files
└── training_history.json          # Training history
```

## 🎯 Cara Penggunaan

1. Buka aplikasi di browser
2. Masukkan teks review di text area, atau klik salah satu contoh review
3. Klik tombol "🔍 Analisis Sentimen"
4. Lihat hasil analisis berupa:
   - Label sentimen (Positif/Netral/Negatif)
   - Tingkat keyakinan model
   - Probabilitas untuk setiap kelas

## 🛠️ Teknologi

- **Model**: IndoBERT (indobenchmark/indobert-base-p1)
- **Framework**: PyTorch
- **Web Framework**: Streamlit
- **Language**: Python 3.11+

## 📊 Screenshot

```
┌─────────────────────────────────────────┐
│     🛵 Analisis Sentimen Review Ojol    │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ Masukkan review di sini...      │   │
│  └─────────────────────────────────┘   │
│                                         │
│        [🔍 Analisis Sentimen]          │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │         😊 POSITIF              │   │
│  │    Tingkat keyakinan: 95.2%     │   │
│  └─────────────────────────────────┘   │
│                                         │
│  Negatif ████░░░░░░░░░ 3.2%           │
│  Netral  █░░░░░░░░░░░░ 1.6%           │
│  Positif ████████████░ 95.2%          │
└─────────────────────────────────────────┘
```

## 📝 Catatan

- Model memerlukan bahasa Indonesia untuk hasil terbaik
- Review yang lebih panjang dan detail akan memberikan hasil lebih akurat
- First load mungkin memerlukan waktu untuk mengunduh model BERT
