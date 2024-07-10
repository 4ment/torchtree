torchtree.distributions.tree_prior
==================================

.. py:module:: torchtree.distributions.tree_prior

.. autoapi-nested-parse::

   Phylogenetic tree priors.



Classes
-------

.. autoapisummary::

   torchtree.distributions.tree_prior.CompoundGammaDirichletPrior


Module Contents
---------------

.. py:class:: CompoundGammaDirichletPrior(id_: torchtree.typing.ID, tree_model: torchtree.evolution.tree_model.UnRootedTreeModel, alpha: torchtree.core.abstractparameter.AbstractParameter, c: torchtree.core.abstractparameter.AbstractParameter, shape: torchtree.core.abstractparameter.AbstractParameter, rate: torchtree.core.abstractparameter.AbstractParameter)

   Bases: :py:obj:`torchtree.core.model.CallableModel`


   Compound gamma-Dirichlet prior on an unrooted tree
   :footcite:t:`rannala2012tail`.

   :param str id_: identifier of object
   :param UnRootedTreeModel tree_model: unrooted tree model
   :param AbstractParameter alpha: concentration parameter of Dirichlet distribution
   :param AbstractParameter c: ratio of the mean internal/external branch lengths
   :param AbstractParameter shape: shape parameter of the gamma distribution
   :param AbstractParameter rate: rate parameter of the gamma distribution

   .. footbibliography::


   .. py:method:: handle_parameter_changed(variable, index, event) -> None


   .. py:method:: from_json(data: dict[str, Any], dic: dict[str, torchtree.core.identifiable.Identifiable]) -> CompoundGammaDirichletPrior
      :classmethod:


      Creates a CompoundGammaDirichletPrior object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a
          CompoundGammaDirichletPrior object.
      :param dict[str, Identifiable] dic: dictionary containing torchtree objects
          keyed by their ID.

      **JSON attributes**:

       Mandatory:
        - id (str): unique string identifier.
        - tree_model (dict or str): a tree model of type
          :class:`~torchtree.evolution.tree_model.UnRootedTreeModel`.
        - alpha (dict or float): concentration parameter of Dirichlet distribution.
        - c (dict or float): ratio of the mean internal/external branch lengths.
        - shape (dict or float): shape parameter of the gamma distribution.
        - rate (dict or float): rate parameter of the gamma distribution.



