torchtree.inference.hmc.hamiltonian
===================================

.. py:module:: torchtree.inference.hmc.hamiltonian


Classes
-------

.. autoapisummary::

   torchtree.inference.hmc.hamiltonian.Hamiltonian


Module Contents
---------------

.. py:class:: Hamiltonian(id_: torchtree.typing.ID, joint: torchtree.core.model.CallableModel)

   Bases: :py:obj:`torchtree.core.model.CallableModel`


   Classes inheriting from :class:`Model` and
   :class:`collections.abc.Callable`.

   CallableModel are Callable and the returned value is cached in case
   we need to use this value multiple times without the need to
   recompute it.


   .. py:attribute:: joint


   .. py:method:: sample_momentum(mass_matrix: torch.Tensor) -> None


   .. py:method:: potential_energy() -> torch.Tensor


   .. py:method:: kinetic_energy(momentum: torch.Tensor, inverse_mass_matrix: torch.Tensor) -> torch.Tensor


   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: from_json(data, dic) -> Hamiltonian
      :classmethod:


      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



