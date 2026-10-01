"""Extract features for the fixed retained cohort using confirmed event labels."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gkan.eeglab import main

if __name__ == "__main__":
    main()
