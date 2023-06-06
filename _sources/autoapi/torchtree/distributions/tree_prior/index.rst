:py:mod:`torchtree.distributions.tree_prior`
============================================

.. py:module:: torchtree.distributions.tree_prior

.. autoapi-nested-parse::

   Phylogenetic tree priors.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.distributions.tree_prior.CompoundGammaDirichletPrior




.. py:class:: CompoundGammaDirichletPrior(id_: torchtree.typing.ID, tree_model: torchtree.evolution.tree_model.UnRootedTreeModel, alpha: torchtree.core.abstractparameter.AbstractParameter, c: torchtree.core.abstractparameter.AbstractParameter, shape: torchtree.core.abstractparameter.AbstractParameter, rate: torchtree.core.abstractparameter.AbstractParameter)

   Bases: :py:obj:`torchtree.core.model.CallableModel`

   Compound gamma-Dirichlet prior on an unrooted tree [rannala2011]_

   :param id_: ID of object
   :param UnRootedTreeModel tree_model: unrooted tree model
   :param AbstractParameter alpha: concentration parameter of Dirichlet distribution
   :param AbstractParameter c: ratio of the mean internal/external branch lengths
   :param AbstractParameter shape: shape parameter of the gamma distribution
   :param AbstractParameter rate: rate parameter of the gamma distribution

   .. [rannala2011]  Rannala, Zhu, and Ziheng Yang. Tail Paradox, Partial
    Identifiability, and Influential Priors in Bayesian Branch Length Inference. 2011

   .. py:method:: handle_parameter_changed(variable, index, event) -> None


   .. py:method:: from_json(data, dic)
      :classmethod:



