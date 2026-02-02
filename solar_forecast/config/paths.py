from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Root project path
PROJ_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Model and output directories
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Config directories
CONFIG_DIR = PROJ_ROOT / "solar_forecast" / "config"
GROUND_DATASET_CONFIG = CONFIG_DIR / "dataset" / "ground.yaml"
SATELLITE_DATASET_CONFIG = CONFIG_DIR / "dataset" / "satellite.yaml"
MODEL_CONFIG = CONFIG_DIR / "dataset" / "model.yaml"

CHECKPOINT_DIR = PROCESSED_DATA_DIR / "checkpoints"
PREDICTIONS_DIR = PROCESSED_DATA_DIR / "predictions"
PLOTS_DIR = PROCESSED_DATA_DIR / "plots"
EVAL_DIR = PROCESSED_DATA_DIR / "evaluation"
