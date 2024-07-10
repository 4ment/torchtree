torchtree.distributions.multivariate_normal
===========================================

.. py:module:: torchtree.distributions.multivariate_normal

.. autoapi-nested-parse::

   Multivariate normal distribution.



Classes
-------

.. autoapisummary::

   torchtree.distributions.multivariate_normal.MultivariateNormal


Module Contents
---------------

.. py:class:: MultivariateNormal(id_: torchtree.typing.ID, x: Union[torchtree.core.abstractparameter.AbstractParameter, list[torchtree.core.abstractparameter.AbstractParameter]], loc: torchtree.core.abstractparameter.AbstractParameter, covariance_matrix: torchtree.core.abstractparameter.AbstractParameter = None, precision_matrix: torchtree.core.abstractparameter.AbstractParameter = None, scale_tril: torchtree.core.abstractparameter.AbstractParameter = None)

   Bases: :py:obj:`torchtree.distributions.distributions.DistributionModel`


   Multivariate normal distribution model.

   :param str or None id_: ID of MultivariateNormal distribution
   :param AbstractParameter or list[AbstractParameter] x: random variable(s) to
       evaluate/sample using distribution.
   :param AbstractParameter loc: mean of the distribution.
   :param AbstractParameter covariance_matrix: covariance of the distribution.
   :param AbstractParameter precision_matrix: precision of the distribution.
   :param AbstractParameter scale_tril: scale tril of the distribution.


   .. py:method:: rsample(sample_shape=torch.Size()) -> None

      Generates a sample_shape shaped reparameterized sample or
      sample_shape shaped batch of reparameterized samples if the
      distribution parameters are batched.



   .. py:method:: sample(sample_shape=torch.Size()) -> None

      Generates a sample_shape shaped sample or sample_shape shaped batch
      of samples if the distribution parameters are batched.



   .. py:method:: log_prob(x: torchtree.Parameter = None) -> torch.Tensor

      Returns the log of the probability density/mass function evaluated
      at x.

      :param Parameter x: value to evaluate
      :return: log probability
      :rtype: Tensor



   .. py:method:: entropy() -> torch.Tensor

      Returns entropy of distribution, batched over batch_shape.

      :return: Tensor of shape batch_shape.
      :rtype: Tensor



   .. py:property:: event_shape
      :type: torch.Size



   .. py:property:: batch_shape
      :type: torch.Size



   .. py:method:: from_json(data, dic)
      :classmethod:


      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



