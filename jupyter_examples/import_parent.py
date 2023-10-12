from pathlib import Path
import sys

parent_dir = str(Path().resolve().parents[0])

sys.path.insert(1, parent_dir)
