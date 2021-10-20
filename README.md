# phylotorch

[![Python package](https://github.com/4ment/phylotorch/actions/workflows/python-package.yml/badge.svg)](https://github.com/4ment/phylotorch/actions/workflows/python-package.yml)

## Installation

### Get the phylotorch source
```bash
git clone https://github.com/4ment/phylotorch
cd phylotorch
```

### Install dependencies

Installing dependencies using Anaconda
```bash
conda env create -f environment.yml
conda activate phylotorch
```

or using pip
```bash
pip install -r requirements.txt
```

### Install phylotorch
```bash
python setup.py install
```

### Check install
```bash
phylotorch --help
```

### Quick start
```bash
phylotorch examples/advi/fluA.json
```