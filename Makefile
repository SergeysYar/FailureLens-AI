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
test:
	pytest -q

