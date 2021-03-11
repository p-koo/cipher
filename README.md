# Cipher

Codebase for the [Koo Lab](https://koolab.cshl.edu/) at Cold Spring Harbor Laboratory.

This is a work in progress.

# Developers

Clone and install the package as below. This also creates a virtual environment, so the dependencies of this package are contained within this directory.

```bash
git clone https://github.com/p-koo/cipher.git
cd cipher
python -m venv venv
source ./venv/bin/activate
python -m pip install --no-cache-dir --upgrade pip setuptools wheel
python -m pip install --no-cache-dir --editable .[dev]
```

## Code formatting

Format your code with `black`.

```bash
black cipher
```

## Running tests

Run tests with `pytest`:

```bash
pytest cipher
```
