# Climate Data

This package contains pipelines and utilities to systematically extract, format, and downscale
data from ERA5 climate models and CMIP6 climate forecasts.

## Developer Installation

Instructions using [`conda`](https://docs.anaconda.com/miniconda/):

1. Clone this repository.

    Over ssh:
    ```sh
    git clone git@github.com:ihmeuw/climate-data.git
    ```

    Over https:
    ```sh
    git clone https://github.com/ihmeuw/climate-data.git
    ```

2. Create a new conda environment.

    ```sh
    conda create -n climate-data python=3.12
    conda activate climate-data
    ```

3. Install `poetry` and the project dependencies.

    ```sh
    pip install poetry
    cd climate-data
    poetry install
    ```

### Pre-commit

[`pre-commit`](https://pre-commit.com/) hooks run all the auto-formatting (`ruff format`),
linters (e.g. `ruff` and `mypy`), and other quality checks to make sure the changeset is
in good shape before a commit/push happens.

You can install the hooks with (runs for each commit):

```sh
pre-commit install
```

Or if you want them to run only for each push:

```sh
pre-commit install -t pre-push
```

Or if you want e.g. want to run all checks manually for all files:

```sh
poetry run pre-commit run --all-files
```

`pre-commit` is configured in the `.pre-commit-config.yaml` file in the repository root.
All auto-formatting, linting, and other tooling is configured in the `pyproject.toml` file.
