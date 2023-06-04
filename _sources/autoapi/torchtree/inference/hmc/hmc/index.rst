:py:mod:`torchtree.inference.hmc.hmc`
=====================================

.. py:module:: torchtree.inference.hmc.hmc


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.inference.hmc.hmc.HMC




.. py:class:: HMC(parameters: torchtree.typing.ListParameter, joint: torchtree.core.model.CallableModel, iterations: int, integrator: torchtree.inference.hmc.integrator.Integrator, **kwargs)



   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: sample_momentum(params)


   .. py:method:: hamiltonian(momentum)


   .. py:method:: run() -> None


   .. py:method:: find_reasonable_step_size()


   .. py:method:: from_json(data: dict[str, any], dic: dict[str, any]) -> HMC
      :classmethod:



