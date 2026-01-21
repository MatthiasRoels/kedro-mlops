lint:
	pre-commit run -a --hook-stage manual $(hook)
test:
	uv run pytest src/tests/

integration-test:
	uv run kedro run --env=test --pipeline=mod_linear_regression_training_pipeline

install-pre-commit:
	pre-commit install --install-hooks

uninstall-pre-commit:
	pre-commit uninstall

secret-scan:
	trufflehog --max_depth 1 --exclude_paths trufflehog-ignore.txt .
