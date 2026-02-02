#!/bin/bash

RUN_ID=$(date +"%Y%m%d_%H%M%S")

# echo "=== Satellite-only ==="
# python -m solar_forecast.train train-satellite \
#   --epochs 50 --batch-size 32 --lr 1e-3 --val-ratio 0.2

# echo "=== Ground-only ==="
# python -m solar_forecast.train train-ground \
#   --epochs 50 --batch-size 32 --lr 1e-3 --val-ratio 0.2

echo "=== Fusion run: $RUN_ID ==="
python -m solar_forecast.train_huber train-fusion \
    --epochs 100 \
    --batch-size 64 \
    --lr 5e-5 \
    --weight-decay 5e-4 \
    --patience 20 \
    --max-grad-norm 1.0 \
    --lr-patience 10 \
    --lr-factor 0.7 \
    --val-ratio 0.15 \
    --ckpt-path data/processed/checkpoints/fusion_best_${RUN_ID}.pt
