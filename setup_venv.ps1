# Script PowerShell untuk Setup Virtual Environment
# Project: Sentiment Analysis Ojek Online Reviews

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SETUP VIRTUAL ENVIRONMENT" -ForegroundColor Cyan
Write-Host "Project: Sentiment Analysis Ojol Review" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Cek Python installation
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Python not found! Please install Python first." -ForegroundColor Red
    exit 1
}

# 2. Buat virtual environment
Write-Host ""
Write-Host "[2/5] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "⚠ Virtual environment already exists. Skipping..." -ForegroundColor Yellow
} else {
    python -m venv venv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Virtual environment created successfully!" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to create virtual environment!" -ForegroundColor Red
        exit 1
    }
}

# 3. Activate virtual environment
Write-Host ""
Write-Host "[3/5] Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
Write-Host "✓ Virtual environment activated!" -ForegroundColor Green

# 4. Upgrade pip
Write-Host ""
Write-Host "[4/5] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
Write-Host "✓ Pip upgraded successfully!" -ForegroundColor Green

# 5. Install requirements
Write-Host ""
Write-Host "[5/5] Installing required packages..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ All packages installed successfully!" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to install some packages!" -ForegroundColor Red
    exit 1
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SETUP COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Virtual environment is already activated" -ForegroundColor White
Write-Host "2. Open Jupyter Notebook dengan: jupyter notebook" -ForegroundColor White
Write-Host "3. Atau buka scraping_playstore_reviews.ipynb di VS Code" -ForegroundColor White
Write-Host ""
Write-Host "To activate venv later, run:" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To deactivate venv, run:" -ForegroundColor Yellow
Write-Host "  deactivate" -ForegroundColor White
Write-Host ""
