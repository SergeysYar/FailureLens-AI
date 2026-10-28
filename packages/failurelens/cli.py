from __future__ import annotations

import argparse
from pathlib import Path

from .demo import write_demo
from .model import train

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "demo" / "telemetry.csv"
ARTIFACT = ROOT / "artifacts" / "baseline.joblib"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["demo", "train"])
    args = parser.parse_args()
    if args.command == "demo":
        print(f"Generated {write_demo(DATA)}")
    else:
        if not DATA.exists(): write_demo(DATA)
        import pandas as pd
        print(train(pd.read_csv(DATA), ARTIFACT))


if __name__ == "__main__":
    main()

