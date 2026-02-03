# Lance Satellite Only
python -m solar_forecast.optuna_satellite_only \
    --n-trials 50 \
    --storage "sqlite:///optuna_satellite.db"

# # Puis Ground Only
# python -m solar_forecast.optuna_ground_only \
#     --n-trials 50 \
#     --storage "sqlite:///optuna_ground.db"


# # Enfin Fusion
# python -m solar_forecast.optuna_fusion \
#     --n-trials 100 \
#     --storage "sqlite:///optuna_fusion.db"