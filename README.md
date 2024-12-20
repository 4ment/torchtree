# torchtree

[![Python package](https://github.com/4ment/torchtree/actions/workflows/python-package.yml/badge.svg)](https://github.com/4ment/torchtree/actions/workflows/python-package.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![docs](https://github.com/4ment/torchtree/actions/workflows/publish_documentation.yml/badge.svg)](https://github.com/4ment/torchtree/actions/workflows/publish_documentation.yml)
![PyPI](https://img.shields.io/pypi/v/torchtree)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchtree)


torchtree is a program designed for developing and inferring phylogenetic models. Implemented in Python, it leverages [PyTorch] for automatic differentiation. The suite of inference algorithms encompasses variational inference, Hamiltonian Monte Carlo, maximum *a posteriori*, and Markov chain Monte Carlo.

For a comprehensive assessment of torchtree's performance and use cases, please see our evaluation repository, [torchtree-experiments](https://github.com/4ment/torchtree-experiments), where torchtree was rigorously tested on various datasets and benchmarked for accuracy and speed.


- [Getting Started](#getting-started)
  - [Dependencies](#dependencies)
  - [Installation](#installation)
- [Quick start](#quick-start)
- [Documentation](#documentation)
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

`torchtree-cli` implements several subcommands, each corresponding to a different type of inference algorithm.
A list of available subcommands can be obtained by running `torchtree-cli --help`.

The following subcommands are available:

* `advi`: Automatic differentiation variational inference
* `hmc`: Hamiltonian Monte Carlo
* `map`: Maximum *a posteriori*
* `mcmc`: Markov chain Monte Carlo

Each subcommand/algorithm requires a different set of arguments which can be obtained by running `torchtree-cli <subcommand> --help`.

`torchtree-cli` requires an alignment file in FASTA format and a tree file in either [Newick](https://en.wikipedia.org/wiki/Newick_format) or [NEXUS](https://en.wikipedia.org/wiki/Nexus_file) format.
While *torchtree* uses the [DendroPy](https://jeetsukumaran.github.io/DendroPy) library to parse and manipulate phylogenetic trees, it is recommended to use a Newick file due to the numerous variations of the NEXUS format.

Let's explore a few examples of how to use these programs using an influenza A virus dataset containing 69 DNA sequences.
The alignment and tree files are located in the [data](data) directory.

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

## How to cite

If you use torchtree, please consider citing:

```

@misc{fourment2024torchtree,
      title={torchtree: flexible phylogenetic model development and inference using {PyTorch}}, 
      author={Mathieu Fourment and Matthew Macaulay and Christiaan J Swanepoel and Xiang Ji and Marc A Suchard and Frederick A Matsen IV},
      year={2024},
      eprint={2406.18044},
      archivePrefix={arXiv},
      primaryClass={q-bio.PE},
      url={https://arxiv.org/abs/2406.18044}
}
```

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