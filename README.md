# kedro-mlops

Opinionated machine learning pipelines for training and inference.

## goal

To do

## Overview

To do.

### How to install dependencies

Declare any dependencies in `pyproject.toml` using `uv add [package-name]`. 
This will also create a virtual environment `.venv` if not exists and installs the package.

To create a virtual environment and install all dependencies, run:

```
uv sync --frozen 
```

Add a `--no-dev` flag to omit installation of dev dependencies.

### How to run your Kedro pipeline

You can run your Kedro project with:

```
uv run kedro run
```

### Comparing gradient boosting packages

Candidate options: xgboost, lightgbm and catboost. We compared the three packages based
on 3 criteria:

1) number of releases in 2022-2023
2) number of dependencies (important when working with containers)
3) package size

Based on those criteria, we had the following result:

|               | xgboost          | lightgbm         | catboost  |
|---------------|------------------|------------------|-----------|
| # releases    | 14               | 7                | 9         |
| dependencies  | 2 (numpy/scipy)  | 2 (numpy/scipy)  | 18        |
| size (MB)     | 6.2              | 5.1              | 90        |
