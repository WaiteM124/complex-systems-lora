PYTHON ?= python3
PIP ?= pip
IMAGE ?= complex-systems-lora:dev
APP_ENV ?= local
export APP_ENV

.PHONY: install lint format typecheck test eval docker-build run \
	generate-dry generate review-export review-import budget \
	train-check baseline-dry baseline judge-dry judge score

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

# Milestones 2-3. Training and local-GPU eval run on Kaggle (see README).
train-check:
	$(PYTHON) -m app.train.qlora --check-data --data data/private/reviewed.jsonl

baseline-dry:
	$(PYTHON) -m app.eval.bedrock_baseline --dry-run

baseline:
	@echo "WARNING: make baseline uses AWS credentials and incurs Bedrock spend (eval phase)."
	$(PYTHON) -m app.eval.bedrock_baseline

judge-dry:
	$(PYTHON) -m app.eval.judge --dry-run

judge:
	@echo "WARNING: make judge uses AWS credentials and incurs Bedrock spend (eval phase)."
	$(PYTHON) -m app.eval.judge

score:
	@test -n "$(RUN)" || (echo "usage: make score RUN=<run-name>"; exit 2)
	$(PYTHON) -m app.eval.score --run $(RUN)
