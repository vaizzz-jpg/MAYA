"""Thin wrapper so ``python scripts/train.py --profile debug`` works."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.training.train import main

if __name__ == "__main__":
    raise SystemExit(main())
