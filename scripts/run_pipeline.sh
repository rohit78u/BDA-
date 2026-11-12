#!/usr/bin/env bash
# Linux / WSL2 pipeline runner
set -e
echo "Running traffic Big Data pipeline in Linux/WSL2..."
source .venv/bin/activate
python -m src.data.prepare_dataset --num-sensors 25
python -m src.training.train_model
python -m src.training.evaluate_model
pytest tests/ -v
echo "Pipeline completed successfully."
