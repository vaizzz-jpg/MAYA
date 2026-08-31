"""Application entrypoint for local development and container deployment.

Usage (from repository root):
    python backend/run.py            # dev default: 127.0.0.1:5000 debug
    python backend/run.py --host 0.0.0.0 --port 5000 --no-debug
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app import create_app  # noqa: E402

app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAYA backend server")
    parser.add_argument(
        "--host", default=os.getenv("MAYA_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("MAYA_PORT", "5000")),
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        default=(os.getenv("MAYA_DEBUG", "1") == "1"),
    )
    parser.add_argument("--no-debug", dest="debug", action="store_false")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
