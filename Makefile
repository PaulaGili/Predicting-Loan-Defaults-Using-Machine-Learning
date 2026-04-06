# needs GNU make — on Windows install via Git for Windows or `choco install make`
.PHONY: install run run-fast test clean

install:
	pip install -r requirements.txt

run:
	python main.py --config config/config.yaml

run-fast:
	python main.py --config config/config.yaml --no-tune

test:
	pytest tests/ -v

clean:
	rm -rf outputs/* models/* logs/*
