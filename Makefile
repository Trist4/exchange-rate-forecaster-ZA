# Convenience targets — every one is a one-liner into run.py / pytest so
# there is exactly one implementation of the pipeline (in Python).
# Assumes the virtualenv is activated (see README quickstart).

.PHONY: install discover fetch build backtest evaluate plots all test

install:
	pip install -r requirements.txt
	# econdatapy lives on TEST PyPI; --no-deps keeps its dependencies
	# coming from real PyPI (already satisfied by requirements.txt).
	pip install -i https://test.pypi.org/simple/ --no-deps econdatapy

# Usage: make discover ID=<dataset-id> [FILTER=EXCX135]
discover:
	python -m src.data.econdata_client $(ID) $(FILTER)

fetch:
	python run.py fetch

build:
	python run.py build

backtest:
	python run.py backtest

evaluate:
	python run.py evaluate

plots:
	python run.py plots

all:
	python run.py

test:
	python -m pytest tests/ -v
