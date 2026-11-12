# Windows PowerShell End-to-End Pipeline Runner
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  REAL-TIME URBAN TRAFFIC PREDICTION & BIG DATA PIPELINE   " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Activate Virtual Environment
Write-Host "`n[1/5] Activating Python 3.11 Virtual Environment..." -ForegroundColor Yellow
$venvPython = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Error: Virtual environment not found. Please run 'uv venv .venv --python 3.11' first." -ForegroundColor Red
    exit 1
}

# 2. Check Dataset and Features
Write-Host "`n[2/5] Checking Dataset and Preprocessed Features..." -ForegroundColor Yellow
if (-not (Test-Path "data/processed/train_features.parquet")) {
    Write-Host "Generating features from METR-LA..." -ForegroundColor Yellow
    & $venvPython -m src.data.prepare_dataset --num-sensors 25
} else {
    Write-Host "Features already prepared. Skipping generation." -ForegroundColor Green
}

# 3. Check Trained Model
Write-Host "`n[3/5] Checking Machine Learning Model..." -ForegroundColor Yellow
if (-not (Test-Path "models/random_forest_traffic.joblib")) {
    Write-Host "Training Baseline and Random Forest Regressor..." -ForegroundColor Yellow
    & $venvPython -m src.training.train_model
    & $venvPython -m src.training.evaluate_model
} else {
    Write-Host "Trained model found at models/random_forest_traffic.joblib." -ForegroundColor Green
}

# 4. Run Pytest Suite
Write-Host "`n[4/5] Executing Automated Test Suite..." -ForegroundColor Yellow
& ".\.venv\Scripts\pytest.exe" tests/ -v
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Some tests failed. Please review test output." -ForegroundColor Red
} else {
    Write-Host "All automated tests passed successfully!" -ForegroundColor Green
}

# 5. Launch Instructions
Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "                 PIPELINE READY FOR DEMO                  " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "To run the components in separate terminal windows:"
Write-Host "1. Replay Kafka stream:   .\.venv\Scripts\python.exe -m src.ingestion.kafka_producer --limit 500 --rate 20"
Write-Host "2. Run Spark Streaming:   .\.venv\Scripts\python.exe -m src.streaming.spark_streaming_job"
Write-Host "3. Launch Dashboard:      .\.venv\Scripts\streamlit.exe run src/dashboard/app.py"
Write-Host "==========================================================" -ForegroundColor Cyan
