lint:
	pre-commit run -a --hook-stage manual $(hook)
test:
	pytest --numprocesses 4 tests/

integration-test:
	kedro run --env=test

install-pre-commit:
	pre-commit install --install-hooks

uninstall-pre-commit:
	pre-commit uninstall
