demo:
	python -m failurelens.cli demo
train:
	python -m failurelens.cli train
experiment:
	python -m failurelens.cli experiment --horizon 7
test:
	pytest -q

