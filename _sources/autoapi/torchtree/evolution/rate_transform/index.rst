torchtree.evolution.rate_transform
==================================

.. py:module:: torchtree.evolution.rate_transform


Classes
-------

.. autoapisummary::

   torchtree.evolution.rate_transform.LogDifferenceRateTransform
   torchtree.evolution.rate_transform.RescaledRateTransform


Module Contents
---------------

.. py:class:: LogDifferenceRateTransform(tree_model: torchtree.evolution.tree_model.TreeModel, cache_size=0)

   Bases: :py:obj:`torch.distributions.Transform`


   Compute log rate difference of adjacent nodes.

   :math:`y_i = \log(r_i) - \log(r_{p(i)})`


   .. py:attribute:: bijective
      :value: True



   .. py:attribute:: sign


   .. py:method:: log_abs_det_jacobian(x, y) -> torch.Tensor

      Computes the log det jacobian `log |dy/dx|` given input and output.



.. py:class:: RescaledRateTransform(rate: torchtree.core.abstractparameter.AbstractParameter, tree_model: torchtree.evolution.tree_model.TreeModel, cache_size=0)

   Bases: :py:obj:`torch.distributions.Transform`


   Scale substitution rates

   :math:`r_i = \mu \tilde{r}_i \frac{\sum b}{\sum b r}`


   .. py:attribute:: bijective
      :value: True



   .. py:attribute:: sign


   .. py:method:: log_abs_det_jacobian(x, y) -> torch.Tensor
      :abstractmethod:


      Computes the log det jacobian `log |dy/dx|` given input and output.



