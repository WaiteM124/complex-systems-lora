PYTHON ?= python3
PIP ?= pip
IMAGE ?= complex-systems-lora:dev
APP_ENV ?= local
export APP_ENV

.PHONY: install lint format typecheck test eval docker-build run \
	generate-dry generate review-export review-import budget

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
	$(PYTHON) -m app sources

generate-dry:
	$(PYTHON) -m app generate --dry-run --fixtures

generate:
	@echo "WARNING: make generate uses AWS credentials and incurs Bedrock spend."
	$(PYTHON) -m app generate

review-export:
	$(PYTHON) -m app export-review

review-import:
	$(PYTHON) -m app import-review

budget:
	$(PYTHON) -m app budget
