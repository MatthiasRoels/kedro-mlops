lint:
	pre-commit run -a --hook-stage manual $(hook)
test:
	pytest --numprocesses 4 tests/

install-pre-commit:
	pre-commit install --install-hooks

uninstall-pre-commit:
	pre-commit uninstall
