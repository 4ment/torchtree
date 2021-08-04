# phylotorch
[![Build Status](https://travis-ci.org/4ment/phylotorch.svg?branch=master)](https://travis-ci.org/4ment/phylotorch)

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