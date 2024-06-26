lint:
	pre-commit run -a --hook-stage manual $(hook)
test:
	pytest src/tests/

integration-test:
	python -m kedro run --env=test --async

install-pre-commit:
	pre-commit install --install-hooks

uninstall-pre-commit:
	pre-commit uninstall

secret-scan:
	trufflehog --max_depth 1 --exclude_paths trufflehog-ignore.txt .
