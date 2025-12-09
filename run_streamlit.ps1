# ============================================
# Run Streamlit Web Application
# ============================================
# Script untuk menjalankan aplikasi web sentiment analysis

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   🚗 Gojek Sentiment Analysis Web App" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if in correct directory
$currentDir = Get-Location
$webDir = Join-Path $currentDir "web"

if (Test-Path $webDir) {
    Write-Host "✓ Found web directory" -ForegroundColor Green
    Set-Location $webDir
} elseif (Test-Path "app.py") {
    Write-Host "✓ Already in web directory" -ForegroundColor Green
} else {
    Write-Host "✗ Error: Cannot find web directory or app.py" -ForegroundColor Red
    Write-Host "Please run this script from the project root directory" -ForegroundColor Yellow
    exit 1
}

# Check if requirements are installed
Write-Host ""
Write-Host "Checking dependencies..." -ForegroundColor Yellow

$pipList = pip list 2>$null
if ($pipList -match "streamlit") {
    Write-Host "✓ Streamlit is installed" -ForegroundColor Green
} else {
    Write-Host "✗ Streamlit not found. Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

# Check if model exists
Write-Host ""
Write-Host "Checking model files..." -ForegroundColor Yellow

$parentDir = Split-Path -Parent (Get-Location)
$model3Class = Join-Path $parentDir "saved_model_indobert_3class"
$model5Class = Join-Path $parentDir "saved_model_indobert_5class"

if (Test-Path $model3Class) {
    Write-Host "✓ Found 3-class model" -ForegroundColor Green
} else {
    Write-Host "⚠ Warning: 3-class model not found at:" -ForegroundColor Yellow
    Write-Host "  $model3Class" -ForegroundColor Gray
    Write-Host "  Please train the model first using Training_IndoBERT_3Class.ipynb" -ForegroundColor Gray
}

if (Test-Path $model5Class) {
    Write-Host "✓ Found 5-class model" -ForegroundColor Green
} else {
    Write-Host "⚠ Note: 5-class model not found (optional)" -ForegroundColor Gray
}

# Run Streamlit
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Streamlit application..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Access the app at: http://localhost:8501" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start Streamlit with proper configuration
streamlit run app.py `
    --server.port 8501 `
    --server.headless false `
    --browser.gatherUsageStats false
