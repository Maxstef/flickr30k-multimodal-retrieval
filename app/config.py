from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

APP_DATA_DIR = PROJECT_ROOT / "app_data"
APP_IMAGES_DIR = APP_DATA_DIR / "images"

MODEL_PATH = PROJECT_ROOT / "models" / "mini_clip.pt"

TOP_K = 5
MATCH_THRESHOLD = 0.25
