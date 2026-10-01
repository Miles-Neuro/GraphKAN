"""Run directly in a terminal or PyCharm; default JSON is Configs/seed_sd.json."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gkan.engine import main

if __name__ == "__main__":
    main()
