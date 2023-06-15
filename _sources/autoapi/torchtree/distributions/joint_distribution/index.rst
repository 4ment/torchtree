:py:mod:`torchtree.distributions.joint_distribution`
====================================================

.. py:module:: torchtree.distributions.joint_distribution

.. autoapi-nested-parse::

   Joint distribution of independent variables.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.distributions.joint_distribution.JointDistributionModel




.. py:class:: JointDistributionModel(id_: torchtree.typing.ID, distributions: list[torchtree.distributions.distributions.DistributionModel])


   Bases: :py:obj:`torchtree.distributions.distributions.DistributionModel`

   Joint distribution of independent variables.

   A JointDistributionModel object contains a list of DistributionModels

   :param id_: ID of joint distribution
   :param distributions: list of distributions of type DistributionModel or
    CallableModel

   .. py:method:: log_prob(x: Union[list[torchtree.Parameter], torchtree.Parameter] = None) -> torch.Tensor

      Returns the log of the probability density/mass function evaluated at x.

      :param Parameter x: value to evaluate
      :return: log probability
      :rtype: Tensor


   .. py:method:: rsample(sample_shape=torch.Size()) -> None

      Generates a sample_shape shaped reparameterized sample or sample_shape
      shaped batch of reparameterized samples if the distribution parameters
      are batched.


   .. py:method:: sample(sample_shape=torch.Size()) -> None

      Generates a sample_shape shaped sample or sample_shape shaped batch of
      samples if the distribution parameters are batched.


   .. py:method:: entropy() -> torch.Tensor

      Returns entropy of distribution, batched over batch_shape.

      :return: Tensor of shape batch_shape.
      :rtype: Tensor


   .. py:method:: handle_parameter_changed(variable: torchtree.Parameter, index, event) -> None


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



