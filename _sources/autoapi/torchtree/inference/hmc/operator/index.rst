:py:mod:`torchtree.inference.hmc.operator`
==========================================

.. py:module:: torchtree.inference.hmc.operator


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.inference.hmc.operator.HMCOperator




.. py:class:: HMCOperator(id_: torchtree.typing.ID, joint: torchtree.core.model.CallableModel, parameters: torchtree.typing.ListParameter, weight: float, target_acceptance_probability: float, integrator: torchtree.inference.hmc.integrator.Integrator, mass_matrix: torchtree.core.abstractparameter.AbstractParameter, adaptors: list[torchtree.inference.hmc.adaptation.Adaptor], **kwargs)

   Bases: :py:obj:`torchtree.inference.mcmc.operator.MCMCOperator`, :py:obj:`torchtree.core.parametric.ParameterListener`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:property:: mass_matrix
      :type: torch.Tensor


   .. py:property:: adaptable_parameter
      :type: torch.Tensor


   .. py:method:: update_mass_matrices() -> None


   .. py:method:: handle_parameter_changed(variable: torchtree.core.abstractparameter.AbstractParameter, index, event) -> None


   .. py:method:: set_adaptable_parameter(value) -> None


   .. py:method:: tune(acceptance_prob: torch.Tensor, sample: int, accepted: bool) -> None


   .. py:method:: from_json(data, dic)
      :classmethod:



