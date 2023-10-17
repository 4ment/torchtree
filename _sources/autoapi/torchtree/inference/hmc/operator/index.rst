:py:mod:`torchtree.inference.hmc.operator`
==========================================

.. py:module:: torchtree.inference.hmc.operator


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.inference.hmc.operator.HMCOperator




.. py:class:: HMCOperator(id_: torchtree.typing.ID, joint: torchtree.core.model.CallableModel, parameters: torchtree.typing.ListParameter, integrator: torchtree.inference.hmc.integrator.Integrator, mass_matrix: torchtree.core.abstractparameter.AbstractParameter, weight: float = 1.0, target_acceptance_probability: float = 0.8, adaptors: list[torchtree.inference.hmc.adaptation.Adaptor] = [], **kwargs)


   Bases: :py:obj:`torchtree.inference.mcmc.operator.MCMCOperator`, :py:obj:`torchtree.core.parametric.ParameterListener`

   Abstract class making an object identifiable.

   :param str or None id_: identifier of object

   .. py:property:: mass_matrix
      :type: torch.Tensor


   .. py:property:: tuning_parameter
      :type: float


   .. py:method:: update_mass_matrices() -> None


   .. py:method:: handle_parameter_changed(variable: torchtree.core.abstractparameter.AbstractParameter, index, event) -> None


   .. py:method:: adaptable_parameter() -> torch.Tensor


   .. py:method:: set_adaptable_parameter(value) -> None


   .. py:method:: tune(acceptance_prob: torch.Tensor, sample: int, accepted: bool) -> None


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



