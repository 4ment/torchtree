:py:mod:`torchtree.nf.flow`
===========================

.. py:module:: torchtree.nf.flow


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.nf.flow.NormalizingFlow




.. py:class:: NormalizingFlow(id_: str, x: Union[torchtree.core.abstractparameter.AbstractParameter, list[torchtree.core.abstractparameter.AbstractParameter]], base: torchtree.distributions.distributions.Distribution, modules: list[torchtree.nn.module.Module], dtype=None, device=None)


   Bases: :py:obj:`torchtree.distributions.distributions.DistributionModel`

   Class for normalizing flows.

   :param id_: ID of object
   :type id_: str or None
   :param x: parameter or list of parameters
   :type x: List[Parameter]
   :param Distribution base: base distribution
   :param modules: list of transformations
   :type modules: List[Module]

   .. py:method:: forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]


   .. py:method:: apply_flow(sample_shape: torch.Size)


   .. py:method:: sample(sample_shape=Size()) -> None

      Generates a sample_shape shaped sample or sample_shape shaped batch
      of samples if the distribution parameters are batched.


   .. py:method:: rsample(sample_shape=Size()) -> None

      Generates a sample_shape shaped reparameterized sample or
      sample_shape shaped batch of reparameterized samples if the
      distribution parameters are batched.


   .. py:method:: log_prob(x: Union[list[torchtree.core.abstractparameter.AbstractParameter], torchtree.core.abstractparameter.AbstractParameter] = None) -> torch.Tensor

      Returns the log of the probability density/mass function evaluated at x.

      :param Parameter x: value to evaluate
      :return: log probability
      :rtype: Tensor


   .. py:method:: entropy() -> torch.Tensor

      Returns entropy of distribution, batched over batch_shape.

      :return: Tensor of shape batch_shape.
      :rtype: Tensor


   .. py:method:: parameters() -> list[torchtree.core.abstractparameter.AbstractParameter]

      Returns parameters of instance Parameter.


   .. py:method:: to(*args, **kwargs) -> None


   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None) -> None


   .. py:method:: cpu() -> None


   .. py:method:: from_json(data: dict[str, any], dic: dict[str, any]) -> NormalizingFlow
      :classmethod:

      Create a Flow object.

      :param data: json representation of Flow object.
      :param dic: dictionary containing additional objects that can be
       referenced in data.

      :return: a :class:`~torchtree.nn.flow.NormalizingFlow` object.
      :rtype: NormalizingFlow



