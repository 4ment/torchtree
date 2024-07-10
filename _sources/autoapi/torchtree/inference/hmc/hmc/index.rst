torchtree.inference.hmc.hmc
===========================

.. py:module:: torchtree.inference.hmc.hmc


Classes
-------

.. autoapisummary::

   torchtree.inference.hmc.hmc.HMC


Module Contents
---------------

.. py:class:: HMC(parameters: torchtree.typing.ListParameter, joint: torchtree.core.model.CallableModel, iterations: int, integrator: torchtree.inference.hmc.integrator.Integrator, **kwargs)

   Bases: :py:obj:`torchtree.core.serializable.JSONSerializable`, :py:obj:`torchtree.core.runnable.Runnable`


   Interface making an object JSON serializable.

   Serializable base class establishing
   :meth:`~torch.core.serializable.JSONSerializable.from_json` abstract method.


   .. py:method:: sample_momentum(params)


   .. py:method:: hamiltonian(momentum)


   .. py:method:: run() -> None


   .. py:method:: find_reasonable_step_size()


   .. py:method:: from_json(data: dict[str, any], dic: dict[str, any]) -> HMC
      :classmethod:


      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



