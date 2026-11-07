import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
vision_dir = root_dir / "vision"

for p in [root_dir, vision_dir]:
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

