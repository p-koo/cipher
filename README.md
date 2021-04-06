# LIBRE -- Learning Interpretable Biological REpresentations

Codebase for the [Koo Lab](https://koolab.cshl.edu/) at Cold Spring Harbor Laboratory.

This is a work in progress.

[![Results of continuous integration](https://github.com/p-koo/libre/workflows/CI/badge.svg)](https://github.com/p-koo/libre/actions/workflows/continuous-integration.yml)
[![Codecov](https://img.shields.io/codecov/c/github/p-koo/libre)](https://app.codecov.io/gh/p-koo/libre/branch/master)

# Install

One can install the master branch (the bleeding edge) with `pip`. One must also install
TensorFlow.

```bash
pip install https://github.com/p-koo/libre/tarball/master tensorflow
```

# Developers

Clone and install the package as below. This also creates a virtual environment, so the
dependencies of this package are contained within this directory.

*Note regarding TensorFlow*: the installation of TensorFlow depends on the environment.
For example, if there is no GPU, one can install `tensorflow-cpu`. If there is a GPU,
then one must take care to install the proper TensorFlow version for the given CUDA
and cuDNN versions.

```bash
git clone https://github.com/p-koo/libre.git
cd libre
python -m venv venv
source ./venv/bin/activate
python -m pip install --no-cache-dir --upgrade pip setuptools wheel
python -m pip install --no-cache-dir --editable .[dev] tensorflow
```

It is also useful to use `pre-commit` to run the styler (`black`) and lint the code on
every commit.

```
pre-commit
```

## Running tests

Run tests with `pytest`:

```bash
pytest libre
```
