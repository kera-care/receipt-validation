.PHONY: test test-data test-verbose

test:
	poetry run pytest tests/

test-data:
	poetry run pytest tests/data/

test-verbose:
	poetry run pytest tests/ -v
