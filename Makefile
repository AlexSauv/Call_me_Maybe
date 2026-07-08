MYPY=mypy
FLK=flake8

.PHONY: run install lint lint-strict

install:
	uv sync

run:
	uv run python -m src $(ARGS)

lint:
	uv run $(FLK) .
	uv run $(MYPY) . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run $(FLK) .
	uv run  $(MYPY) --strict . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs