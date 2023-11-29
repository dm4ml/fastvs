.PHONY: tests lint release install

tests:
	poetry run pytest

lint:
	ruff . --fix

release:
	poetry version patch
	poetry publish --build

install:
	pip install poetry
	poetry install

build:
	poetry run maturin develop --release