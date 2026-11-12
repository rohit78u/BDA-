# Start Kafka and Kafka UI using Docker Compose
Write-Host "Starting Kafka Broker and Kafka UI containers..." -ForegroundColor Cyan
docker compose up -d
Write-Host "Waiting for Kafka to be healthy..." -ForegroundColor Yellow
Start-Sleep -Seconds 5
docker compose ps
Write-Host "Kafka UI available at: http://localhost:8080" -ForegroundColor Green
