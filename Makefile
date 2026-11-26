PYTHON ?= python3
PIP ?= pip
IMAGE ?= ai-project:dev
APP_ENV ?= local
export APP_ENV

.PHONY: install lint format typecheck test eval docker-build run

install:
	$(PIP) install -e ".[dev]"
	pre-commit install

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .

typecheck:
	mypy src

test:
	pytest -q

eval:
	$(PYTHON) evals/run_offline.py

docker-build:
	docker build -t $(IMAGE) .

run:
	$(PYTHON) -m app
