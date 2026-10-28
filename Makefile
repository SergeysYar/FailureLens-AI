demo:
	python -m failurelens.cli demo
train:
	python -m failurelens.cli train
experiment:
	python -m failurelens.cli experiment --horizon 7
research-quick:
	python -m failurelens.cli research-quick --horizon 7
research-full:
	python -m failurelens.cli research-full --horizon 7
optimize:
	python -m failurelens.cli optimize
evaluate:
	python -m failurelens.cli evaluate --horizon 7
backtest:
	python -m failurelens.cli backtest --horizon 7
research-report:
	python -m failurelens.cli research-report --horizon 7
data-validate:
	python -m failurelens.cli data-validate
features:
	python -m failurelens.cli features --horizon 7
register-model:
	python -m failurelens.cli register-model --horizon 7
promote-model:
	python -m failurelens.cli promote-model --version $(VERSION)
rollback:
	python -m failurelens.cli rollback --version $(VERSION)
retrain:
	python -m failurelens.cli retrain --horizon 7
benchmark:
	python -m failurelens.cli benchmark --horizon 7
monitor-drift:
	python -m failurelens.cli monitor-drift
lint:
	ruff check packages tests
test:
	pytest -q

