from __future__ import annotations

import argparse
from pathlib import Path

from .demo import write_demo
from .evaluation import run_experiments
from .model import train

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "demo" / "telemetry.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["demo", "train", "experiment"])
    parser.add_argument("--horizon", type=int, choices=[7, 30], default=7)
    args = parser.parse_args()
    if args.command == "demo":
        print(f"Generated {write_demo(DATA)}")
    elif args.command == "train":
        if not DATA.exists(): write_demo(DATA)
        import pandas as pd
        artifact = ROOT / "artifacts" / f"baseline_{args.horizon}d.joblib"
        print(train(pd.read_csv(DATA), artifact, horizon=args.horizon))
    elif args.command == "experiment":
        if not DATA.exists(): write_demo(DATA)
        import json

        import pandas as pd
        result = run_experiments(pd.read_csv(DATA), horizon=args.horizon)
        target = ROOT / "artifacts/experiments" / f"horizon_{args.horizon}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

