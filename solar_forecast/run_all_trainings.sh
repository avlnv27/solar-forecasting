#!/bin/bash

echo "=== Satellite-only ==="
python -m solar_forecast.train train-satellite \
  --epochs 300 --batch-size 32 --lr 1e-3 --val-ratio 0.2

echo "=== Ground-only ==="
python -m solar_forecast.train train-ground \
  --epochs 400 --batch-size 32 --lr 1e-3 --val-ratio 0.2

echo "=== Fusion ==="
python -m solar_forecast.train train-fusion \
  --epochs 500 --batch-size 32 --lr 1e-3 --val-ratio 0.2
