:py:mod:`torchtree.inference.mcmc.mcmc`
=======================================

.. py:module:: torchtree.inference.mcmc.mcmc


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.inference.mcmc.mcmc.MCMC




.. py:class:: MCMC(joint: torchtree.core.model.CallableModel, operators: List[torchtree.inference.mcmc.operator.MCMCOperator], iterations: int, **kwargs)

   Bases: :py:obj:`torchtree.core.serializable.JSONSerializable`, :py:obj:`torchtree.core.runnable.Runnable`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: run() -> None


   .. py:method:: from_json(data: dict[str, any], dic: dict[str, any]) -> MCMC
      :classmethod:



