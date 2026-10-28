#!/bin/sh
set -eu
if [ "${FAILURELENS_BOOTSTRAP_DEMO:-0}" = "1" ] && [ ! -f artifacts/baseline_7d.joblib ]; then
  python -m failurelens.cli demo
  python -m failurelens.cli train --horizon 7
  python -m failurelens.cli train --horizon 30
fi
exec "$@"
