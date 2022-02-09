# torchtree

[![Python package](https://github.com/4ment/torchtree/actions/workflows/python-package.yml/badge.svg)](https://github.com/4ment/torchtree/actions/workflows/python-package.yml)

## Installation

### Get the torchtree source
```bash
git clone https://github.com/4ment/torchtree
cd torchtree
```

### Install dependencies

Installing dependencies using Anaconda
```bash
conda env create -f environment.yml
conda activate torchtree
```

or using pip
```bash
pip install -r requirements.txt
```

### Install torchtree
```bash
python -m pip install .
```

### Check install
```bash
torchtree --help
```

### Quick start
```bash
torchtree examples/advi/fluA.json
```