:py:mod:`torchtree.inference.mcmc.operator`
===========================================

.. py:module:: torchtree.inference.mcmc.operator


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.inference.mcmc.operator.MCMCOperator
   torchtree.inference.mcmc.operator.ScalerOperator
   torchtree.inference.mcmc.operator.SlidingWindowOperator
   torchtree.inference.mcmc.operator.DirichletOperator




.. py:class:: MCMCOperator(id_: torchtree.typing.ID, parameters: list[torchtree.typing.Parameter], weight: float, target_acceptance_probability: float, **kwargs)


   Bases: :py:obj:`torchtree.core.identifiable.Identifiable`, :py:obj:`abc.ABC`

   Abstract class making an object identifiable.

   :param str or None id_: identifier of object

   .. py:property:: tuning_parameter
      :type: float
      :abstractmethod:


   .. py:property:: adaptable_parameter
      :type: float
      :abstractmethod:


   .. py:method:: set_adaptable_parameter(value: float) -> None
      :abstractmethod:


   .. py:method:: step() -> torch.Tensor


   .. py:method:: accept() -> None


   .. py:method:: reject() -> None


   .. py:method:: tune(acceptance_prob: torch.Tensor, sample: int, accepted: bool) -> None


   .. py:method:: state_dict() -> dict[str, Any]


   .. py:method:: load_state_dict(state_dict: dict[str, Any]) -> None



.. py:class:: ScalerOperator(id_: torchtree.typing.ID, parameters: list[torchtree.typing.Parameter], weight: float, target_acceptance_probability: float, scaler: float, **kwargs)


   Bases: :py:obj:`MCMCOperator`

   Abstract class making an object identifiable.

   :param str or None id_: identifier of object

   .. py:property:: tuning_parameter
      :type: float


   .. py:method:: adaptable_parameter() -> float


   .. py:method:: set_adaptable_parameter(value: float) -> None


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: SlidingWindowOperator(id_: torchtree.typing.ID, parameters: list[torchtree.typing.Parameter], weight: float, target_acceptance_probability: float, width: float, **kwargs)


   Bases: :py:obj:`MCMCOperator`

   Abstract class making an object identifiable.

   :param str or None id_: identifier of object

   .. py:property:: tuning_parameter
      :type: float


   .. py:method:: adaptable_parameter() -> float


   .. py:method:: set_adaptable_parameter(value: float) -> None


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: DirichletOperator(id_: torchtree.typing.ID, parameters: torchtree.typing.Parameter, weight: float, target_acceptance_probability: float, scaler: float, **kwargs)


   Bases: :py:obj:`MCMCOperator`

   Abstract class making an object identifiable.

   :param str or None id_: identifier of object

   .. py:property:: tuning_parameter
      :type: float


   .. py:method:: adaptable_parameter() -> float


   .. py:method:: set_adaptable_parameter(value: float) -> None


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



