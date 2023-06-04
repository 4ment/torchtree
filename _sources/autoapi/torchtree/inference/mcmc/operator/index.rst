:py:mod:`torchtree.inference.mcmc.operator`
===========================================

.. py:module:: torchtree.inference.mcmc.operator


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.inference.mcmc.operator.MCMCOperator




.. py:class:: MCMCOperator(id_: torchtree.typing.ID, joint: torchtree.core.model.CallableModel, parameters: list[torchtree.typing.Parameter], weight: float, target_acceptance_probability: float, **kwargs)



   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:property:: adaptable_parameter
      :type: float
      :abstractmethod:


   .. py:method:: set_adaptable_parameter(value: float) -> None
      :abstractmethod:


   .. py:method:: step() -> torch.Tensor


   .. py:method:: accept() -> None


   .. py:method:: reject() -> None


   .. py:method:: tune(acceptance_prob: torch.Tensor, sample: int, accepted: bool) -> None



