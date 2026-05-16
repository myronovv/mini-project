from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

FEATURE_COLUMNS = [
    "voltage_kv",
    "current_a",
    "frequency_hz",
    "active_power_mw",
    "reactive_power_mvar",
    "transformer_temp_c",
    "line_load_pct",
]

RANDOM_SEED = 42
LATENT_DIM = 8
HIDDEN_DIMS = (32, 16)
EPOCHS = 80
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
THRESHOLD_PERCENTILE = 95.0
