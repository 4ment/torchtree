:py:mod:`torchtree.distributions.gmrf`
======================================

.. py:module:: torchtree.distributions.gmrf


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.distributions.gmrf.GMRF
   torchtree.distributions.gmrf.GMRFCovariate




.. py:class:: GMRF(id_: torchtree.typing.ID, field: torchtree.core.abstractparameter.AbstractParameter, precision: torchtree.core.abstractparameter.AbstractParameter, tree_model: torchtree.evolution.tree_model.TimeTreeModel = None, weights: torch.Tensor = None, rescale: bool = True)



   Classes inheriting from :class:`Model` and
   :class:`collections.abc.Callable`.

   CallableModel are Callable and the returned value is cached in case
   we need to use this value multiple times without the need to
   recompute it.

   .. py:method:: from_json(data, dic)
      :classmethod:



.. py:class:: GMRFCovariate(id_: torchtree.typing.ID, field: torchtree.core.abstractparameter.AbstractParameter, precision: torchtree.core.abstractparameter.AbstractParameter, covariates: torchtree.core.abstractparameter.AbstractParameter, beta: torchtree.core.abstractparameter.AbstractParameter)



   Classes inheriting from :class:`Model` and
   :class:`collections.abc.Callable`.

   CallableModel are Callable and the returned value is cached in case
   we need to use this value multiple times without the need to
   recompute it.

   .. py:method:: from_json(data, dic)
      :classmethod:



