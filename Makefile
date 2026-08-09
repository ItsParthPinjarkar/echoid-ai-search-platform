.PHONY: install run test lint format eval docker-up docker-down

install:
	pip install -e ".[dev]"

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest -v

lint:
	ruff check .
	mypy app

format:
	ruff format .

eval:
	python -m eval.run_eval

docker-up:
	docker compose up --build

docker-down:
	docker compose down
