.PHONY: run test lint
run:
	.venv/bin/uvicorn app.main:app --port 8001 --reload

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check app tests
