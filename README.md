# torchtree

[![Python package](https://github.com/4ment/torchtree/actions/workflows/python-package.yml/badge.svg)](https://github.com/4ment/torchtree/actions/workflows/python-package.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![docs](https://github.com/4ment/torchtree/actions/workflows/publish_documentation.yml/badge.svg)](https://github.com/4ment/torchtree/actions/workflows/publish_documentation.yml)
![PyPI](https://img.shields.io/pypi/v/torchtree)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchtree)


torchtree is a program designed for developing and inferring phylogenetic models. Implemented in Python, it leverages [PyTorch] for automatic differentiation. The suite of inference algorithms encompasses variational inference, Hamiltonian Monte Carlo, maximum *a posteriori*, and Markov chain Monte Carlo.

- [Getting Started](#getting-started)
  - [Dependencies](#dependencies)
  - [Installation](#installation)
- [Quick start](#quick-start)
- [Documentatoin](#documentation)
- [Plug-ins](#torchtree-plug-in)

## Getting Started

### Dependencies
 - [DendroPy]
 - [PyTorch]

 ### Installation
Use an Anaconda environment (Optional)
```bash
conda env create -f environment.yml
conda activate torchtree
```

To install the latest stable version you can run
```bash
pip install torchtree
```

To build torchtree from source you can run
```bash
git clone https://github.com/4ment/torchtree
pip install torchtree/
```

Check install
```bash
torchtree --help
```

## Documentation
For detailed information on how to use `torchtree` and its features, please refer to the official documentation and API reference.
 - [Documentation](https://4ment.github.io/torchtree)
 - [API Reference](https://4ment.github.io/torchtree/autoapi/torchtree/index.html)

## Quick start
`torchtree` requires a JSON file containing models and algorithms. A configuration file can be generated using `torchtree-cli`, a command line-based tool. This two-step process allows the user to adjust values in the configuration file, such as hyperparameters.

### 1 - Generating a configuration file
Some examples of models using variational inference:

#### Unrooted tree with GTR+W4 model
*W4* refers to a site model with 4 rates categories coming from a discretized Weibull distribution. This is similar to the more commonly used discretized Gamma distribution site model.

```bash
torchtree-cli advi -i data/fluA.fa -t data/fluA.tree -m GTR -C 4 > fluA.json
```

#### Time tree with strict clock and constant coalescent model
```bash
torchtree-cli advi -i data/fluA.fa -t data/fluA.tree -m JC69 --clock strict --coalescent constant > fluA.json
```

### 2 - Running torchtree
This will generate `sample.csv` and `sample.trees` files containing parameter and tree samples drawn from the variational distribution
```bash
torchtree fluA.json
```

## torchtree plug-in
torchtree can be easily extended without modifying the code base thanks its modular implementation. Some examples of plug-ins:

- [torchtree-bito]
- [torchtree-physher]
- [torchtree-scipy]
- [torchtree-tensorflow]

A GitHub [template](https://github.com/4ment/torchtree-plugin-template) is available to assist in the development of a plug-in, and it is highly recommended to use it. This template provides a structured starting point, ensuring consistency and compatibility with `torchtree` while streamlining the development process.

## License

Distributed under the GPLv3 License. See [LICENSE](LICENSE) for more information.

## Acknowledgements

torchtree makes use of the following libraries and tools, which are under their own respective licenses:

 - [PyTorch]
 - [DendroPy]

[DendroPy]: https://github.com/jeetsukumaran/DendroPy
[PyTorch]: https://pytorch.org
[torchtree-bito]: https://github.com/4ment/torchtree-bito
[torchtree-physher]: https://github.com/4ment/torchtree-physher
[torchtree-scipy]: https://github.com/4ment/torchtree-scipy
[torchtree-tensorflow]: https://github.com/4ment/torchtree-tensorflow