# Script untuk menjalankan Streamlit Web App
# Jalankan dari root folder project

Write-Host "🛵 Starting Sentiment Analysis Web App..." -ForegroundColor Cyan
Write-Host ""

# Check if streamlit is installed
$streamlitCheck = pip show streamlit 2>$null
if (-not $streamlitCheck) {
    Write-Host "⚠️ Streamlit belum terinstall. Menginstall..." -ForegroundColor Yellow
    pip install streamlit
}

# Check if model exists
$modelPath = "models/indobert_sentiment_3class.pt"
if (-not (Test-Path $modelPath)) {
    Write-Host "⚠️ Model belum tersedia di: $modelPath" -ForegroundColor Yellow
    Write-Host "   Jalankan notebook 'sentiment_training_3class_final.ipynb' terlebih dahulu" -ForegroundColor Yellow
    Write-Host ""
}

# Run streamlit
Write-Host "🚀 Starting Streamlit server..." -ForegroundColor Green
Write-Host "   URL: http://localhost:8501" -ForegroundColor White
Write-Host ""

streamlit run web/app.py
