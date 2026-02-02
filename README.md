# SolarForecasting

<!-- <a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project. -->

## Repository organization

```
SOLAR-FORECASTING/
│
├── solar_forecast/                
│   │
│   ├── config/
│   │   ├── dataset/
│   │   │   ├── ground.yaml        # Ground data configuration
│   │   │   ├── satellite.yaml     # Satellite data configuration
│   │   │   └── model.yaml         # Model architecture & hyperparameters
│   │   │
│   │   ├── logging.py             # Logging configuration
│   │   └── paths.py               # Project paths
│   │
│   ├── nn/
│   │   ├── layers/                # Custom neural network layers
│   │   │   ├── attention.py
│   │   │   ├── sampling_readout.py
│   │   │   └── __init__.py
│   │   │
│   │   ├── models/                     # Model definitions
│   │   │   ├── fusion.py               # Satellite + Ground Fusion model
│   │   │   ├── satellite_only.py       # CNN satellite-only baseline
│   │   │   ├── ground_only.py          # GNN ground-only baseline
│   │   │   ├── satellite_cnn.py        # Satellite CNN backbone
│   │   │   └── time_then_graph_iso.py  # Ground GNN backbone
│   │   │
│   │   ├── utils.py               # Dataset, DataLoader, helpers for preprocessing
│   │   ├── load_data.py
│   │   └── __init__.py
│   │
│   ├── load_data.py               # Data loading (process .nc files)
│   ├── preprocess_data.py         # Data cleaning & preprocessing (CSI index, night time removed...)
│   ├── train.py                   # MSE training loop
│   ├── train_huber.py             # Huber training loop
│   ├── predict.py                 # Multi-horizon prediction script
│   └── run_all_trainings.sh       # Shell script to launch training runs for the 3 models
│
├── data/
│   ├── raw/                       # Raw input data
│   │   ├── ground/
│   │   └── satellite/
│   │
│   ├── interim/
│   │   ├── ground/
│   │   │   └── *_processed.csv
│   │   └── satellite/
│   │       └── satellite_irradiance.csv
│   │
│   └── processed/
│       ├── ground/
│       │   └── *_processed_clean.csv
│       │
│       ├── satellite/
│       │   └── satellite_irradiance_clean.csv
│       │
│       ├── checkpoints/
│       │   ├── fusion_best.pt
│       │   ├── fusion_best_*.pt        # Saved runs
│       │   ├── satellite_only_best.pt
│       │   └── ground_only_best.pt
│       │
│       ├── predictions/
│       │   ├── fusion_predictions.csv
│       │   ├── satellite_only_predictions.csv
│       │   ├── ground_only_predictions.csv
│       │   └── persistence_predictions.csv
│       │
│       └── evaluation/
│           ├── metrics_by_horizon.png
│           ├── normalized_metrics_by_horizon.png
│           ├── residual_analysis.png
│           ├── scatter_pred_vs_true.png
│           └── ...
│
├── notebooks/
│   └── analysis_notebook.ipynb     # Evaluation & visualization notebook
│
├── models/                         # (optional) exported / external models
├── reports/                        # Figures, tables, report material
├── references/                     # Papers, references
│
├── requirements.txt
├── pyproject.toml
├── Makefile
└── README.md

```

--------

