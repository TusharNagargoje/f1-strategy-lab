from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models"
CACHE_DIR = DATA_DIR / "fastf1_cache"

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"

# Compound ordering used across the project (soft = fastest, hard = slowest/most durable)
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

# Laps we drop when fitting pace models (first lap, in/out laps, SC/VSC/red-flag laps)
RANDOM_SEED = 42


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROC_DIR, MODEL_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
