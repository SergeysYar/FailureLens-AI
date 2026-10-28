from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .demo import write_demo
from .evaluation import run_experiments
from .features import BASE, make_features
from .model import train
from .monitoring import drift_report
from .registry import register, transition
from .research import run_advanced, write_report

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "demo" / "telemetry.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["demo", "train", "experiment", "research-quick", "research-full", "evaluate", "backtest", "research-report", "optimize", "data-validate", "features", "register-model", "promote-model", "rollback", "retrain", "benchmark", "monitor-drift"])
    parser.add_argument("--horizon", type=int, choices=[7, 30], default=7)
    parser.add_argument("--version")
    parser.add_argument("--state", choices=["candidate", "staging", "production", "archived"], default="candidate")
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
    elif args.command in {"data-validate", "features", "monitor-drift"}:
        if not DATA.exists(): write_demo(DATA)
        import pandas as pd
        frame = pd.read_csv(DATA)
        if args.command == "data-validate":
            required = {"date", "device_id", *BASE}
            missing = sorted(required - set(frame.columns))
            duplicates = int(frame.duplicated(["device_id", "date"]).sum())
            result = {"status": "valid" if not missing and not duplicates else "invalid", "rows": len(frame), "missing_columns": missing, "duplicate_device_dates": duplicates}
        elif args.command == "features":
            values, _ = make_features(frame, horizon=args.horizon)
            result = {"status": "executed", "rows": len(values), "features": len(values.columns), "schema_version": "temporal-v3"}
        else:
            midpoint = len(frame) // 2
            result = drift_report(frame.iloc[:midpoint], frame.iloc[midpoint:], BASE)
        print(json.dumps(result, indent=2))
    elif args.command in {"register-model", "retrain"}:
        if not DATA.exists(): write_demo(DATA)
        import pandas as pd
        artifact = ROOT / "artifacts" / f"baseline_{args.horizon}d.joblib"
        if args.command == "retrain":
            print(train(pd.read_csv(DATA), artifact, horizon=args.horizon))
        metadata_path = artifact.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        record = register(ROOT / "artifacts/registry.json", artifact, {"algorithm": "logistic_regression", "horizon_days": args.horizon, "feature_schema_version": "temporal-v3", "metrics": metadata}, version=args.version)
        print(json.dumps(record, indent=2))
    elif args.command in {"promote-model", "rollback"}:
        if not args.version:
            raise SystemExit("--version is required")
        state = "production"
        print(json.dumps(transition(ROOT / "artifacts/registry.json", args.version, state), indent=2))
    elif args.command == "benchmark":
        if not DATA.exists(): write_demo(DATA)
        import pandas as pd
        frame = pd.read_csv(DATA).head(240)
        started = time.perf_counter()
        values, _ = make_features(frame, horizon=args.horizon)
        result = {"status": "executed", "rows": len(frame), "features": len(values.columns), "feature_seconds": round(time.perf_counter() - started, 6), "conditions": "local process; demo data; single run"}
        target = ROOT / "artifacts/benchmark.json"
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

