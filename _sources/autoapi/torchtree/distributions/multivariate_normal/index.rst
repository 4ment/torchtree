:py:mod:`torchtree.distributions.multivariate_normal`
=====================================================

.. py:module:: torchtree.distributions.multivariate_normal


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.distributions.multivariate_normal.MultivariateNormal




.. py:class:: MultivariateNormal(id_: torchtree.typing.ID, x: torchtree.core.abstractparameter.AbstractParameter, loc: torchtree.core.abstractparameter.AbstractParameter, covariance_matrix=None, precision_matrix=None, scale_tril=None)



   Multivariate normal distribution.

   :param id_: ID of joint distribution
   :param x: random variable to evaluate/sample using distribution
   :param loc: mean of the distribution
   :param covariance_matrix: covariance matrix Parameter
   :param precision_matrixs: precision matrix Parameter
   :param scale_tril: scale tril Parameter

   .. py:property:: event_shape
      :type: torch.Size


   .. py:property:: batch_shape
      :type: torch.Size


   .. py:method:: rsample(sample_shape=torch.Size()) -> None


   .. py:method:: sample(sample_shape=torch.Size()) -> None


   .. py:method:: log_prob(x: torchtree.Parameter = None) -> torch.Tensor


   .. py:method:: entropy() -> torch.Tensor


   .. py:method:: from_json(data, dic)
      :classmethod:



