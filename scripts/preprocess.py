from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gkan.preprocessing import prepare_main

if __name__ == "__main__":
    prepare_main()
