#!/usr/bin/env bash
# Linux / WSL2 service stopper
set -e
echo "Stopping local infrastructure..."
docker compose down
echo "Containers stopped."
