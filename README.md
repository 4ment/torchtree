# torchtree

[![Python package](https://github.com/4ment/torchtree/actions/workflows/python-package.yml/badge.svg)](https://github.com/4ment/torchtree/actions/workflows/python-package.yml)

## Installation

### Use an Anaconda environment (Optional)
```bash
conda env create -f environment.yml
conda activate torchtree
```

### The easy way
To install the latest stable version, run
```bash
pip install torchtree
```

### Using the source code
```bash
git clone https://github.com/4ment/torchtree
cd torchtree
pip install .
```

## Check install
```bash
torchtree --help
```

## Quick start
torchtree will approximate the posterior distribution of an unrooted tree with a JC69 substitution model using variational inference 
```bash
torchtree examples/advi/fluA.json
```

The JSON file can be generated using the torchtree CLI
```bash
torchtree-cli advi -i data/fluA.fa -t data/fluA.tree > fluA.json
```