#!/usr/bin/env bash
# Linux / WSL2 service starter
set -e
echo "Starting local infrastructure via Docker Compose..."
docker compose up -d
echo "Kafka running on port 9092. Kafka UI on http://localhost:8080"
