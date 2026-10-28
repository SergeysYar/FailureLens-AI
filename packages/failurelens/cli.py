from __future__ import annotations

import argparse
import json
from pathlib import Path

from .demo import write_demo
from .evaluation import run_experiments
from .model import train
from .research import run_advanced, write_report

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "demo" / "telemetry.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["demo", "train", "experiment", "research-quick", "research-full", "evaluate", "backtest", "research-report", "optimize"])
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
        import pandas as pd
        result = run_experiments(pd.read_csv(DATA), horizon=args.horizon)
        target = ROOT / "artifacts/experiments" / f"horizon_{args.horizon}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    elif args.command in {"research-quick", "research-full", "evaluate", "backtest", "research-report"}:
        if not DATA.exists(): write_demo(DATA)
        import pandas as pd
        result = run_advanced(pd.read_csv(DATA), horizon=args.horizon, quick=args.command == "research-quick")
        paths = write_report(result, ROOT / "artifacts/research")
        print(json.dumps({"result": result, "reports": [str(path) for path in paths]}, indent=2))
    elif args.command == "optimize":
        print(json.dumps({"status": "not_executed", "reason": "Optuna is optional and is not installed in the minimal environment."}))


if __name__ == "__main__":
    main()

