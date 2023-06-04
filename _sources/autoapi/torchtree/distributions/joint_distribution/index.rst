:py:mod:`torchtree.distributions.joint_distribution`
====================================================

.. py:module:: torchtree.distributions.joint_distribution


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.distributions.joint_distribution.JointDistributionModel




.. py:class:: JointDistributionModel(id_: torchtree.typing.ID, distributions: list[torchtree.distributions.distributions.DistributionModel])



   Joint distribution of independent distributions.

   :param id_: ID of joint distribution
   :param distributions: list of distributions of type DistributionModel or
    CallableModel

   .. py:method:: log_prob(x: Union[list[torchtree.Parameter], torchtree.Parameter] = None) -> torch.Tensor


   .. py:method:: rsample(sample_shape=torch.Size()) -> None


   .. py:method:: sample(sample_shape=torch.Size()) -> None


   .. py:method:: entropy() -> torch.Tensor


   .. py:method:: handle_parameter_changed(variable: torchtree.Parameter, index, event) -> None


   .. py:method:: from_json(data, dic)
      :classmethod:



