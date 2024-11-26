Welcome to torchtree!
======================================

.. warning::
   The documentation corresponds to the current state of the main branch. There may be differences with the latest released version.


torchtree is a program designed for developing and inferring phylogenetic models.
It is implemented in Python and uses PyTorch to leverage automatic differentiation.
Inference algorithms include variational inference, Hamiltonian Monte Carlo, maximum *a posteriori* and Markov chain Monte Carlo.

For a comprehensive assessment of torchtree's performance and use cases, please see our evaluation repository, `torchtree-experiments <https://github.com/4ment/torchtree-experiments>`_, where torchtree was rigorously tested on various datasets and benchmarked for accuracy and speed.

Installation
------------

.. tabs::

   .. code-tab:: bash Latest

      git clone https://github.com/4ment/torchtree
      pip install torchtree/

   .. code-tab:: bash Pypi

      pip install torchtree


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/install.rst
   getting_started/quick_start.rst
   getting_started/features.rst
   getting_started/json_reference.rst
   getting_started/plugins.rst


.. toctree::
   :maxdepth: 1
   :caption: Advanced

   advanced/concepts.rst
   advanced/tree_likelihood.rst
   advanced/benchmark.rst


How to cite
-----------

.. tabs::

   .. tab:: Cite

      Mathieu Fourment, Matthew Macaulay, Christiaan J Swanepoel, Xiang Ji, Marc A Suchard and Frederick A Matsen IV. *torchtree: flexible phylogenetic model development and inference using PyTorch*, 2024 `arXiv:2406.18044 <https://arxiv.org/abs/2406.18044>`_

   .. code-tab:: bibtex

      @misc{fourment2024torchtree,
            title={torchtree: flexible phylogenetic model development and inference using {PyTorch}}, 
            author={Mathieu Fourment and Matthew Macaulay and Christiaan J Swanepoel and Xiang Ji and Marc A Suchard and Frederick A Matsen IV},
            year={2024},
            eprint={2406.18044},
            archivePrefix={arXiv},
            primaryClass={q-bio.PE},
            url={https://arxiv.org/abs/2406.18044}
      }

.. toctree::
   :hidden:
   :caption: API

   API Reference<autoapi/torchtree/index>
   bibliography/bib
