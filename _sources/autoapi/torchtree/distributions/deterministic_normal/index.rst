:py:mod:`torchtree.distributions.deterministic_normal`
======================================================

.. py:module:: torchtree.distributions.deterministic_normal


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.distributions.deterministic_normal.DeterministicNormal




.. py:class:: DeterministicNormal(id_: Optional[str], loc: torchtree.core.abstractparameter.AbstractParameter, scale: torchtree.core.abstractparameter.AbstractParameter, x: Union[list[torchtree.core.abstractparameter.AbstractParameter], torchtree.core.abstractparameter.AbstractParameter], shape: torch.Size)

   Bases: :py:obj:`torchtree.distributions.distributions.DistributionModel`

   Deterministic Normal distribution.

   Standard normal variates are drawn during object creation and samples drawn
   from this distribution are a transformation of these variates

   :param id_: ID of joint distribution
   :param x: random variable to evaluate/sample using distribution
   :param loc: location of the distribution
   :param scale: scale of the distribution
   :param shape: shape of standard normal variates

   .. py:property:: event_shape
      :type: torch.Size


   .. py:property:: batch_shape
      :type: torch.Size


   .. py:method:: rsample(sample_shape=torch.Size()) -> None

      Generates a sample_shape shaped reparameterized sample or sample_shape
      shaped batch of reparameterized samples if the distribution parameters
      are batched.


   .. py:method:: sample(sample_shape=torch.Size()) -> None

      Generates a sample_shape shaped sample or sample_shape shaped batch of
      samples if the distribution parameters are batched.


   .. py:method:: log_prob(x: Union[list[torchtree.core.abstractparameter.AbstractParameter], torchtree.core.abstractparameter.AbstractParameter] = None) -> torch.Tensor

      Returns the log of the probability density/mass function evaluated at x.

      :param Parameter x: value to evaluate
      :return: log probability
      :rtype: Tensor


   .. py:method:: entropy() -> torch.Tensor

      Returns entropy of distribution, batched over batch_shape.

      :return: Tensor of shape batch_shape.
      :rtype: Tensor


   .. py:method:: handle_model_changed(model: torchtree.core.model.Model, obj, index) -> None


   .. py:method:: json_factory(id_: str, loc: Union[str, dict], scale: Union[str, dict], x: Union[str, dict], shape: list) -> dict
      :staticmethod:


   .. py:method:: from_json(data, dic)
      :classmethod:



