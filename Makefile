lint:
	pre-commit run -a --hook-stage manual $(hook)
test:
	pytest tests/

integration-test:
	kedro run --env=test

install-pre-commit:
	pre-commit install --install-hooks

uninstall-pre-commit:
	pre-commit uninstall

secret-scan:
	trufflehog --max_depth 1 --exclude_paths trufflehog-ignore.txt .
