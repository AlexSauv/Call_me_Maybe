MYPY=mypy
FLK=flake8

.PHONY: run install lint lint-strict debug clean

install:
	uv sync

run:
	uv run python -m src $(ARGS)

debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache
	rm -rf data/output
	find . -type d -name "__pycache__" -exec rm -rf {} +

lint:
	uv run $(FLK) .
	uv run $(MYPY) . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run $(FLK) .
	uv run  $(MYPY) . --strict --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs