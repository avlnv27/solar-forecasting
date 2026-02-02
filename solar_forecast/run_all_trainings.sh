#!/bin/bash

RUN_ID=$(date +"%Y%m%d_%H%M%S")

# echo "=== Satellite-only ==="
# python -m solar_forecast.train train-satellite \
#   --epochs 50 --batch-size 32 --lr 1e-3 --val-ratio 0.2

# python -m solar_forecast.train train-ground \
#   --epochs 50 \
#   --batch-size 32 \
#   --lr 3e-5 \
#   --weight-decay 5e-4 \
#   --val-ratio 0.2 \
#   --patience 20

echo "=== Fusion run: $RUN_ID ==="
python -m solar_forecast.train_huber_randomday \
    --epochs 100 \
    --batch-size 32 \
    --lr 3e-5 \
    --weight-decay 1e-3 \
    --patience 10 \
    --max-grad-norm 1.0 \
    --lr-patience 5 \
    --lr-factor 0.5 \
    --val-ratio 0.15 \
    --ckpt-path data/processed/checkpoints/fusion_best_randday_${RUN_ID}.pt



