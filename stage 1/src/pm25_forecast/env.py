from pathlib import Path
from dotenv import load_dotenv

# Load .env from repo root when this module is imported in notebooks/scripts
repo_root = Path(__file__).resolve().parents[2]
env_path = repo_root / ".env"
if env_path.exists():
    load_dotenv(env_path.as_posix())
