#!/bin/bash

# ============================================================
# Ground Only Model - Complete Pipeline
# ============================================================
# 
# This script runs the complete pipeline for Ground Only:
# 1. Optuna hyperparameter search (50 trials)
# 2. Retrain with best parameters (200 epochs)
# 3. Generate predictions
#
# Usage:
#   chmod +x solar_forecast/run_ground_pipeline.sh
#   ./solar_forecast/run_ground_pipeline.sh [optuna|retrain|predict|all]
#   ./solar_forecast/run_ground_pipeline.sh all
# ============================================================

set -e

# test Configuration
# MODEL_NAME="Ground Only"
# N_TRIALS=1
# MAX_EPOCHS_OPTUNA=1
# PATIENCE_OPTUNA=15
# EPOCHS_RETRAIN=1
# PATIENCE_RETRAIN=25
# STORAGE="sqlite:///optuna_ground.db"
# OUTPUT_DIR="models/best_optuna_ground"

# # Configuration
MODEL_NAME="Ground Only"
N_TRIALS=40
MAX_EPOCHS_OPTUNA=100
PATIENCE_OPTUNA=15
EPOCHS_RETRAIN=150
PATIENCE_RETRAIN=25
STORAGE="sqlite:///optuna_ground.db"
OUTPUT_DIR="models/best_optuna_ground"

# ============================================================
# Helper Functions
# ============================================================

print_header() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
    echo ""
}

# ============================================================
# Check Environment
# ============================================================

check_environment() {
    print_header "Checking Environment"
    
    if [ ! -f "solar_forecast/__init__.py" ]; then
        echo "[ERROR] Not in project root directory"
        exit 1
    fi
    echo "[OK] Project directory"
    
    if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
        echo "[OK] Conda environment: $CONDA_DEFAULT_ENV"
    fi
    
    python -c "import solar_forecast" 2>/dev/null || {
        echo "[ERROR] solar_forecast package not found"
        exit 1
    }
    echo "[OK] solar_forecast package"
    
    GPU_STATUS=$(python -c "import torch; print('GPU' if torch.cuda.is_available() else 'CPU')" 2>/dev/null)
    echo "      Device: $GPU_STATUS"
    echo ""
}

# ============================================================
# Pipeline Steps
# ============================================================

run_optuna() {
    print_header "STEP 1: Optuna Hyperparameter Search"
    echo "  Model: $MODEL_NAME"
    echo "  Trials: $N_TRIALS"
    echo "  Max epochs per trial: $MAX_EPOCHS_OPTUNA"
    echo "  Storage: $STORAGE"
    echo "  Estimated time: 4-6 hours (CPU)"
    echo ""
    
    START_TIME=$(date +%s)
    
    python -m solar_forecast.optuna_ground_only \
        --n-trials $N_TRIALS \
        --max-epochs $MAX_EPOCHS_OPTUNA \
        --patience $PATIENCE_OPTUNA \
        --storage "$STORAGE"
    
    if [ $? -eq 0 ]; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        HOURS=$((DURATION / 3600))
        MINUTES=$(((DURATION % 3600) / 60))
        
        echo ""
        echo "[OK] Optuna search completed"
        echo "     Time: ${HOURS}h ${MINUTES}m"
        echo "     Results: $OUTPUT_DIR/"
        echo ""
        
        if [ -f "$OUTPUT_DIR/best_params.yaml" ]; then
            echo "     Best parameters:"
            cat "$OUTPUT_DIR/best_params.yaml" | head -10
            echo ""
        fi
    else
        echo "[ERROR] Optuna search failed"
        exit 1
    fi
}

run_retrain() {
    print_header "STEP 2: Retrain with Best Parameters"
    
    if [ ! -f "$OUTPUT_DIR/best_params.yaml" ]; then
        echo "[ERROR] Best parameters not found"
        echo "        Run: ./run_ground_pipeline.sh optuna"
        exit 1
    fi
    
    echo "  Model: $MODEL_NAME"
    echo "  Epochs: $EPOCHS_RETRAIN"
    echo "  Patience: $PATIENCE_RETRAIN"
    echo "  Loading params from: $OUTPUT_DIR/best_params.yaml"
    echo ""
    
    START_TIME=$(date +%s)
    
    python -m solar_forecast.retrain_ground_best \
        --epochs $EPOCHS_RETRAIN \
        --patience $PATIENCE_RETRAIN
    
    if [ $? -eq 0 ]; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        MINUTES=$((DURATION / 60))
        
        echo ""
        echo "[OK] Retraining completed"
        echo "     Time: ${MINUTES} minutes"
        echo "     Model saved: $OUTPUT_DIR/best_model.pt"
        echo ""
    else
        echo "[ERROR] Retraining failed"
        exit 1
    fi
}

run_predict() {
    print_header "STEP 3: Generate Predictions"
    
    if [ ! -f "$OUTPUT_DIR/best_model.pt" ]; then
        echo "[ERROR] Trained model not found"
        echo "        Run: ./run_ground_pipeline.sh retrain"
        exit 1
    fi
    
    echo "  Model: $MODEL_NAME"
    echo "  Loading from: $OUTPUT_DIR/best_model.pt"
    echo ""
    
    START_TIME=$(date +%s)
    
    python -m solar_forecast.predict
    
    if [ $? -eq 0 ]; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        MINUTES=$((DURATION / 60))
        
        echo ""
        echo "[OK] Predictions generated"
        echo "     Time: ${MINUTES} minutes"
        echo "     Output: data/processed/predictions/ground_only_predictions.csv"
        echo ""
    else
        echo "[ERROR] Prediction failed"
        exit 1
    fi
}

# ============================================================
# Complete Pipeline
# ============================================================

run_all() {
    PIPELINE_START=$(date +%s)
    
    echo ""
    echo "Running COMPLETE pipeline for $MODEL_NAME"
    echo "  1. Optuna Search (~4-6h CPU)"
    echo "  2. Retrain (~20-40min)"
    echo "  3. Predictions (~5-10min)"
    echo ""
    echo "Total estimated time: 5-7 hours (CPU)"
    echo ""
    read -p "Continue? [y/N] " -n 1 -r
    echo ""
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled"
        exit 0
    fi
    
    run_optuna
    run_retrain
    run_predict
    
    PIPELINE_END=$(date +%s)
    PIPELINE_DURATION=$((PIPELINE_END - PIPELINE_START))
    PIPELINE_HOURS=$((PIPELINE_DURATION / 3600))
    PIPELINE_MINUTES=$(((PIPELINE_DURATION % 3600) / 60))
    
    print_header "Pipeline Completed"
    echo "[OK] Total time: ${PIPELINE_HOURS}h ${PIPELINE_MINUTES}m"
    echo ""
    echo "Results:"
    echo "  - Best params:    $OUTPUT_DIR/best_params.yaml"
    echo "  - Trained model:  $OUTPUT_DIR/best_model.pt"
    echo "  - Predictions:    data/processed/predictions/ground_only_predictions.csv"
    echo "  - Training curve: $OUTPUT_DIR/training_curve.png"
    echo ""
}

# ============================================================
# Main
# ============================================================

show_usage() {
    echo ""
    echo "Usage: ./run_ground_pipeline.sh [command]"
    echo ""
    echo "Commands:"
    echo "  all       Run complete pipeline (Optuna -> Retrain -> Predict)"
    echo "  optuna    Run only Optuna search"
    echo "  retrain   Run only retraining"
    echo "  predict   Run only predictions"
    echo "  help      Show this help"
    echo ""
}

main() {
    echo ""
    echo "================================================================"
    echo "  Ground Only Model Pipeline"
    echo "================================================================"
    echo ""
    
    check_environment
    
    case "${1:-help}" in
        all)
            run_all
            ;;
        optuna)
            run_optuna
            ;;
        retrain)
            run_retrain
            ;;
        predict)
            run_predict
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            echo "[ERROR] Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

trap 'echo ""; echo "[INTERRUPTED]"; exit 130' INT

main "$@"
