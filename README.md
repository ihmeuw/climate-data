# climate-data

---

**Documentation**: [https://ihmeuw.github.io/climate-data](https://ihmeuw.github.io/climate-data)
**Source Code**: [https://github.com/ihmeuw/climate-data](https://github.com/ihmeuw/climate-data)

---

Pipelines to extract, format, and downscale ERA5 and CMIP6 data.


## Development

Instructions using conda:

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
    conda create -n climate-data python=3.11
    conda activate climate-data
    ```

3. Install `poetry` and the project dependencies.

    ```sh
    pip install poetry
    cd climate-data
    poetry install
    ```

### Pre-commit

Pre-commit hooks run all the auto-formatting (`ruff format`), linters
(e.g. `ruff` and `mypy`), and other quality checks to make sure the changeset is in
good shape before a commit/push happens.

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
