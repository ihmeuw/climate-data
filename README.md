# climate-downscale

[![PyPI](https://img.shields.io/pypi/v/climate-downscale?style=flat-square)](https://pypi.python.org/pypi/climate-downscale/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/climate-downscale?style=flat-square)](https://pypi.python.org/pypi/climate-downscale/)
[![PyPI - License](https://img.shields.io/pypi/l/climate-downscale?style=flat-square)](https://pypi.python.org/pypi/climate-downscale/)

---

**Documentation**: [https://.github.io/climate-downscale](https://.github.io/climate-downscale)

**Source Code**: [https://github.com/climate-downscale](https://github.com//climate-downscale)

**PyPI**: [https://pypi.org/project/climate-downscale/](https://pypi.org/project/climate-downscale/)

---

Pipelines to downscale ERA5 and CMIP6 data.


## Development

Instructions using conda:

1. Clone this repository. 

    Over ssh:
    ```sh
    git clone git@github.com:ihmeuw/climate-downscale.git
    ```
    
    Over https:
    ```sh
    git clone https://github.com/ihmeuw/climate-downscale.git
    ```

2. Create a new conda environment.

    ```sh
    conda create -n climate-downscale python=3.10
    conda activate climate-downscale
    ```

3. Install `poetry` and the project dependencies.

    ```sh
    conda install poetry
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
