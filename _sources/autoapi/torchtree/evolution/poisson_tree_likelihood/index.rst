torchtree.evolution.poisson_tree_likelihood
===========================================

.. py:module:: torchtree.evolution.poisson_tree_likelihood


Classes
-------

.. autoapisummary::

   torchtree.evolution.poisson_tree_likelihood.PoissonTreeLikelihood


Module Contents
---------------

.. py:class:: PoissonTreeLikelihood(id_: torchtree.typing.ID, tree_model: torchtree.evolution.tree_model.TimeTreeModel, clock_model: torchtree.evolution.branch_model.BranchModel, edge_lengths: torchtree.core.abstractparameter.AbstractParameter)

   Bases: :py:obj:`torchtree.core.model.CallableModel`


   Tree likelihood class using Poisson model.

   :param id_: ID of object.
   :type id_: str or None
   :param TimeTreeModel tree_model: a tree model.
   :param BranchModel clock_model: a clock model.
   :param Parameter edge_lengths: edge lengths.


   .. py:attribute:: tree_model


   .. py:attribute:: clock_model


   .. py:attribute:: edge_lengths


   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: from_json(data, dic) -> PoissonTreeLikelihood
      :classmethod:


      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



