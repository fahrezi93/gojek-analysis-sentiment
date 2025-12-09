# 📚 Panduan Lengkap Aplikasi Web Sentiment Analysis

## 🎯 Overview

Aplikasi web ini adalah sistem analisis sentimen lengkap untuk ulasan Gojek yang dibangun dengan arsitektur modular dan OOP yang terstruktur.

## 🏗️ Arsitektur Aplikasi

```
web/
├── app.py                 # SentimentUI - Main application & UI logic
├── predictor.py          # ModelPredictor - Model inference engine
├── preprocessing.py      # TextNormalizer - Text cleaning & normalization
├── evaluation.py         # PerformanceEvaluator - Model evaluation metrics
├── requirements.txt      # Python dependencies
└── README.md            # Documentation
```

## 🔄 Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INTERFACE                          │
│                     (SentimentUI)                            │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ├──► Load Model
                  │    └──► ModelPredictor.load_model()
                  │
                  ├──► Analyze Text
                  │    ├──► TextNormalizer.clean_text()
                  │    └──► ModelPredictor.predict_single()
                  │
                  ├──► Analyze CSV
                  │    ├──► TextNormalizer.preprocess_batch()
                  │    └──► ModelPredictor.predict_batch()
                  │
                  └──► Evaluate Model
                       ├──► TextNormalizer.preprocess_batch()
                       ├──► ModelPredictor.predict_batch()
                       └──► PerformanceEvaluator.calculate_metrics()
```

## 📱 Halaman-Halaman Aplikasi

### 1. 🏠 Beranda (Home)

**Fitur:**
- Hero section dengan branding Gojek
- Overview fitur utama (3 cards)
- Quick stats performa model (4 metrics)
- Getting started guide

**Komponen UI:**
- Gradient header dengan warna Gojek (#00AA13)
- Metric cards dengan hover effects
- Information box dengan step-by-step guide

### 2. 🔮 Analisis Sentimen

#### A. Input Manual (Teks)

**Flow:**
1. User ketik/paste teks ulasan
2. Click "ANALISIS SENTIMEN"
3. System:
   - Clean text (TextNormalizer)
   - Predict sentiment (ModelPredictor)
   - Display results

**Output:**
- Result card dengan gradient background
- Emoji sesuai sentimen
- Confidence score (%)
- Tabel distribusi probabilitas
- Expandable section untuk detail preprocessing

**Contoh:**
```
Input: "Aplikasi sangat membantu, driver ramah dan cepat!"

Output:
┌─────────────────────────────────┐
│         😊                       │
│   Sentimen: POSITIF             │
│   Confidence: 98.5%             │
└─────────────────────────────────┘

Probabilitas:
Positif:  98.5% ████████████████████
Netral:    1.2% █
Negatif:   0.3% 
```

#### B. Upload CSV (Batch)

**Flow:**
1. User upload CSV file
2. System preview data (first 10 rows)
3. Auto-detect text column
4. Click "ANALISIS SEMUA DATA"
5. System:
   - Preprocess all texts
   - Batch prediction
   - Generate summary & results

**Output:**
- Summary cards per sentimen (metrics)
- Detailed results table
- Download button untuk CSV hasil

**Format CSV Input:**
```csv
text
"Pelayanan sangat memuaskan"
"Driver tidak profesional"
"Standar, tidak ada yang istimewa"
```

**Format CSV Output:**
```csv
Text Original,Text Cleaned,Prediksi,Confidence (%)
"Pelayanan sangat memuaskan","pelayanan sangat memuaskan","Positif","98.50"
```

### 3. 📊 Evaluasi Model

**Flow:**
1. User upload CSV dengan ground truth labels
2. System validate columns (text, label)
3. Click "MULAI EVALUASI"
4. System:
   - Preprocess texts
   - Predict all
   - Calculate metrics (PerformanceEvaluator)
   - Generate visualizations

**Output:**

#### A. Metrik Utama (4 metrics)
```
┌──────────┬──────────┬──────────┬──────────┐
│ Akurasi  │ Presisi  │ Recall   │F1-Score │
│   92%    │   91%    │   90%    │   91%    │
└──────────┴──────────┴──────────┴──────────┘
```

#### B. Visualisasi
1. **Metrik Per Kelas** (Grouped Bar Chart)
   - Presisi, Recall, F1-Score per kelas
   - Interactive Plotly chart

2. **Distribusi Prediksi** (Pie Chart)
   - Persentase prediksi per kelas
   - Donut chart dengan labels

3. **Confusion Matrix** (Heatmap)
   - Interactive heatmap
   - Annotations dengan jumlah sampel
   - Color scale: Blues

#### C. Detail Tables
- Metrik per kelas (tabel)
- Classification report (text expandable)

**Format CSV Input:**
```csv
text,label
"Aplikasi sangat membantu, driver ramah",2
"Pelayanan buruk, lama banget",0
"Biasa saja, tidak ada yang spesial",1
```

### 4. ℹ️ Tentang

**Sections:**
1. Tentang Proyek
2. Teknologi yang Digunakan (3 columns)
3. Arsitektur Model
4. Fitur Aplikasi (6 feature cards)
5. Dataset Info
6. Developer Credit

## 🎨 Design System

### Color Palette

```css
/* Primary Colors */
--gojek-green: #00AA13;
--gojek-dark: #008f0f;

/* Sentiment Colors */
--positive: #10B981;  /* Green */
--neutral: #FFC107;   /* Yellow */
--negative: #DC3545;  /* Red */

/* UI Colors */
--info: #2196F3;
--background: #f0f2f6;
--card-bg: #ffffff;
--text-primary: #333333;
--text-secondary: #666666;
```

### Typography

```css
/* Headers */
.main-header {
  font-size: 2.5rem;
  font-weight: bold;
  color: #00AA13;
}

.sub-header {
  font-size: 1.5rem;
  font-weight: bold;
  color: #333;
}

/* Body Text */
body {
  font-family: 'Inter', sans-serif;
  font-size: 1rem;
  line-height: 1.5;
}
```

### Components

#### Metric Card
```html
<div class='metric-card'>
  <h3>🔮 Title</h3>
  <p>Description text</p>
</div>
```

#### Info Box
```html
<div class='info-box'>
  <b>Label:</b> Content
</div>
```

#### Result Card (Gradient)
```html
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'>
  <h2>😊</h2>
  <h3>Sentimen: POSITIF</h3>
  <h4>Confidence: 98.5%</h4>
</div>
```

## 🔧 Technical Details

### Session State Management

```python
st.session_state:
├── predictor: ModelPredictor instance
├── normalizer: TextNormalizer instance
├── model_loaded: bool
├── model_type: "3class" | "5class"
├── prediction_results: DataFrame
└── evaluation_results: dict
```

### Model Loading Strategy

1. User selects model type (3-class/5-class)
2. Click "Load Model" button
3. System:
   - Initialize ModelPredictor with type
   - Load tokenizer (IndoBERT)
   - Load model weights from saved_model_*
   - Move to device (GPU/CPU)
   - Set to eval mode
4. Store in session_state
5. Update UI status indicator

### Text Processing Pipeline

```python
Input Text
    ↓
TextNormalizer.clean_text()
    ├── remove_noise()
    │   ├── lowercase
    │   ├── remove URLs
    │   ├── remove mentions
    │   ├── remove hashtags
    │   ├── remove emojis
    │   └── remove punctuation
    │
    ├── normalize_slang()
    │   └── replace with dictionary
    │
    └── whitespace cleanup
    ↓
Cleaned Text
    ↓
Tokenizer (IndoBERT)
    ↓
Model Prediction
    ↓
Results
```

### Batch Processing

```python
for batch in chunks(texts, batch_size=16):
    inputs = tokenizer(batch, ...)
    with torch.no_grad():
        outputs = model(**inputs)
    results.extend(process_outputs(outputs))
```

### Evaluation Metrics Calculation

```python
PerformanceEvaluator:
├── calculate_metrics(y_true, y_pred)
│   ├── accuracy_score()
│   ├── precision_score(weighted)
│   ├── recall_score(weighted)
│   ├── f1_score(weighted)
│   └── confusion_matrix()
│
├── create_confusion_matrix_plot()
│   └── Plotly Heatmap
│
├── create_per_class_metrics_chart()
│   └── Plotly Grouped Bar Chart
│
└── create_prediction_distribution()
    └── Plotly Pie Chart
```

## 🚀 Performance Optimization

### Caching
- Model loading: Cache in session_state
- TextNormalizer: Compiled regex patterns
- Batch processing: Efficient tensor operations

### Memory Management
- Batch size: 16 (adjustable)
- torch.no_grad() for inference
- Clear CUDA cache if needed

### UI Responsiveness
- Progress spinners for long operations
- Chunked processing for large datasets
- Lazy loading for visualizations

## 🐛 Common Issues & Solutions

### Issue 1: Model Not Found
**Solution:**
1. Check path: `saved_model_indobert_3class/`
2. Verify files exist:
   - config.json
   - model.safetensors
   - vocab.txt
3. Train model using notebook if missing

### Issue 2: CUDA Out of Memory
**Solution:**
1. Reduce batch_size in predictor.py
2. Use CPU instead: Remove .cuda() calls
3. Process in smaller chunks

### Issue 3: CSV Upload Error
**Solution:**
1. Check encoding: UTF-8
2. Verify column names: 'text' or 'review'
3. Remove special characters in column names
4. Check for missing values

### Issue 4: Slow Performance
**Solutions:**
1. Use GPU if available
2. Increase batch size (if memory allows)
3. Cache repeated predictions
4. Reduce max_length in tokenizer

## 📊 Example Usage Scenarios

### Scenario 1: Single Review Analysis
```
User: "Saya mau analisis review ini: 'Mantap aplikasinya!'"
System: 
1. Clean text: "mantap aplikasinya"
2. Predict: POSITIF (95.2%)
3. Show probabilities: Positif 95.2%, Netral 3.5%, Negatif 1.3%
```

### Scenario 2: Bulk Analysis (1000 reviews)
```
User: Upload CSV dengan 1000 reviews
System:
1. Process in batches of 16
2. Time: ~30 seconds on GPU
3. Results: 750 Positif, 150 Netral, 100 Negatif
4. Export to CSV
```

### Scenario 3: Model Evaluation (Test Set)
```
User: Upload test set dengan 500 labeled reviews
System:
1. Predict all 500
2. Calculate metrics:
   - Accuracy: 92%
   - Precision: 91%
   - Recall: 90%
   - F1: 91%
3. Generate confusion matrix
4. Show per-class performance
```

## 🔐 Security Considerations

1. **File Upload:**
   - Validate file type (CSV only)
   - Limit file size
   - Sanitize input data

2. **Input Validation:**
   - Check for SQL injection patterns
   - Limit text length
   - Escape special characters

3. **Error Handling:**
   - Catch all exceptions
   - Display user-friendly messages
   - Log errors for debugging

## 📈 Future Enhancements

1. **Multi-language Support:**
   - English reviews
   - Mixed language detection

2. **Advanced Analytics:**
   - Trend analysis over time
   - Topic modeling
   - Aspect-based sentiment

3. **API Integration:**
   - REST API for predictions
   - Webhook support
   - Batch API endpoint

4. **User Management:**
   - Authentication
   - Save analysis history
   - Custom model training

5. **Export Options:**
   - PDF reports
   - Excel format
   - API export

---

**Last Updated:** December 2024
**Version:** 1.0.0
