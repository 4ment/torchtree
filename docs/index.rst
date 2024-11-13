Welcome to torchtree!
======================================

.. warning::
   The documentation corresponds to the current state of the main branch. There may be differences with the latest released version.


torchtree is a program designed for developing and inferring phylogenetic models.
It is implemented in Python and uses PyTorch to leverage automatic differentiation.
Inference algorithms include variational inference, Hamiltonian Monte Carlo, maximum *a posteriori* and Markov chain Monte Carlo.

For a comprehensive assessment of torchtree's performance and use cases, please see our evaluation repository, [torchtree-experiments](https://github.com/4ment/torchtree-experiments), where torchtree was rigorously tested on various datasets and benchmarked for accuracy and speed.

Installation
------------

.. tabs::

   .. code-tab:: bash Latest

      git clone https://github.com/4ment/torchtree
      pip install torchtree/

   .. code-tab:: bash Pypi

      pip install torchtree


Plug-ins
------------------

torchtree can be easily extended without modifying the code base thanks its modular implementation. Some examples of external packages:

- torchtree-bito_: is a plug-in interfacing the bito_ library for fast gradient calculations with BEAGLE_.
- torchtree-physher_: is a plug-in interfacing physher_ for fast gradient calculations of tree likelihood and coalescent models.
- torchtree-scipy_: is a plug-in interfacing the SciPy package.
- torchtree-tensorflow_: is a plug-in interacting with Tensorflow.

A GitHub `template <https://github.com/4ment/torchtree-plugin-template>`_ is available to assist in the development of a plug-in, and it is highly recommended to use it. This template provides a structured starting point, ensuring consistency and compatibility with `torchtree` while streamlining the development process.

.. _torchtree-bito: http://github.com/4ment/torchtree-bito
.. _torchtree-physher: http://github.com/4ment/torchtree-physher
.. _torchtree-scipy: http://github.com/4ment/torchtree-scipy
.. _torchtree-tensorflow: http://github.com/4ment/torchtree-tensorflow
.. _physher: http://github.com/4ment/physher
.. _BEAGLE: https://github.com/beagle-dev/beagle-lib
.. _bito: https://github.com/phylovi/bito


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
   :maxdepth: 1
   :caption: Getting Started

   getting_started/install.rst
   getting_started/quick_start.rst
   getting_started/features.rst
   getting_started/json_reference.rst


.. toctree::
   :maxdepth: 1
   :caption: Advanced

   advanced/concepts.rst
   advanced/tree_likelihood.rst
   advanced/benchmark.rst

.. toctree::
   :hidden:
   :caption: API

   API Reference<autoapi/torchtree/index>
   bibliography/bib
