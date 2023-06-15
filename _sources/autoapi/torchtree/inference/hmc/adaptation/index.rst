:py:mod:`torchtree.inference.hmc.adaptation`
============================================

.. py:module:: torchtree.inference.hmc.adaptation


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.inference.hmc.adaptation.Adaptor
   torchtree.inference.hmc.adaptation.AdaptiveStepSize
   torchtree.inference.hmc.adaptation.DualAveragingStepSize
   torchtree.inference.hmc.adaptation.MassMatrixAdaptor
   torchtree.inference.hmc.adaptation.WarmupAdaptation



Functions
~~~~~~~~~

.. autoapisummary::

   torchtree.inference.hmc.adaptation.find_reasonable_step_size



.. py:class:: Adaptor


   Bases: :py:obj:`torchtree.core.serializable.JSONSerializable`, :py:obj:`abc.ABC`

   Interface making an object JSON serializable.

   Serializable base class establishing
   :meth:`~torch.core.serializable.JSONSerializable.from_json` abstract method.

   .. py:method:: learn(acceptance_prob: torch.Tensor, sample: int, accepted: bool) -> None
      :abstractmethod:


   .. py:method:: restart() -> None
      :abstractmethod:



.. py:class:: AdaptiveStepSize(integrator: torchtree.inference.hmc.integrator.LeapfrogIntegrator, target_acceptance_probability: float, **kwargs)


   Bases: :py:obj:`Adaptor`

   Interface making an object JSON serializable.

   Serializable base class establishing
   :meth:`~torch.core.serializable.JSONSerializable.from_json` abstract method.

   .. py:method:: restart() -> None


   .. py:method:: learn(acceptance_prob: torch.Tensor, sample: int, accepted: bool) -> None


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: DualAveragingStepSize(integrator: torchtree.inference.hmc.integrator.LeapfrogIntegrator, mu=0.5, delta=0.8, gamma=0.05, kappa=0.75, t0=10, **kwargs)


   Bases: :py:obj:`Adaptor`

   Step size adaptation using dual averaging Nesterov.

   Code adapted from: https://github.com/stan-dev/stan

   .. py:method:: restart() -> None


   .. py:method:: learn(acceptance_prob: torch.Tensor, sample: int, accepted: bool) -> None


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: MassMatrixAdaptor(parameters: torchtree.typing.ListParameter, mass_matrix: torchtree.core.abstractparameter.AbstractParameter, regularize=True, **kwargs)


   Bases: :py:obj:`Adaptor`

   Interface making an object JSON serializable.

   Serializable base class establishing
   :meth:`~torch.core.serializable.JSONSerializable.from_json` abstract method.

   .. py:property:: mass_matrix


   .. py:method:: learn(acceptance_prob: torch.Tensor, sample: int, accepted: bool) -> None


   .. py:method:: restart() -> None


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:function:: find_reasonable_step_size(integrator, parameters, hamiltonian, mass_matrix, inverse_mass_matrix)


.. py:class:: WarmupAdaptation


   Bases: :py:obj:`Adaptor`

   Interface making an object JSON serializable.

   Serializable base class establishing
   :meth:`~torch.core.serializable.JSONSerializable.from_json` abstract method.

   .. py:property:: step_size
      :abstractmethod:


   .. py:property:: mass_matrix
      :abstractmethod:


   .. py:property:: inverse_mass_matrix
      :abstractmethod:


   .. py:property:: sqrt_mass_matrix
      :abstractmethod:



