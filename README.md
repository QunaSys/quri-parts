# quri-sdk-notebooks
Example and tutorial notebooks used in QURI SDK documentation

## Installation instructions

The notebooks here use `quri-algo`, `quri-parts`, `quri-vm` and `quri-parts-qsci`. Make sure you have the latest versions of them installed. The easiest way is to use [poetry](https://python-poetry.org/docs/#installing-with-pipx).

```
poetry install
```

Then run the notebooks in your virtual environment. You can run jupyter using

```
poetry run jupyter notebook
```

and then open the jupyter server in your browser.

## Contributing

We require that all contributed notebooks can run without throwing any error and that no metadata is included. You can clean your notebook of metadata by running the command

```
make clean-nb
```

Before opening a pull request make sure that your notebook can run in its entirety without throwing any errors. You can do so using

```
make execute-in-place
```

Beware that this executes any notebooks you have added or modified in place and overwrites output cells. If you only want to check them you can run

```
poetry run jupyter execute quri_sdk_notebooks/path/to/notebook.ipynb
```