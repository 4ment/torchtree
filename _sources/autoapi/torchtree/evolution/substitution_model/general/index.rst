:py:mod:`torchtree.evolution.substitution_model.general`
========================================================

.. py:module:: torchtree.evolution.substitution_model.general


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.evolution.substitution_model.general.GeneralJC69
   torchtree.evolution.substitution_model.general.GeneralSymmetricSubstitutionModel
   torchtree.evolution.substitution_model.general.GeneralNonSymmetricSubstitutionModel
   torchtree.evolution.substitution_model.general.EmpiricalSubstitutionModel




.. py:class:: GeneralJC69(id_: torchtree.typing.ID, state_count: int)

   Bases: :py:obj:`torchtree.evolution.substitution_model.abstract.SubstitutionModel`

   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:property:: frequencies
      :type: torch.Tensor


   .. py:method:: handle_model_changed(model, obj, index)


   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None) -> None


   .. py:method:: cpu() -> None


   .. py:method:: p_t(branch_lengths: torch.Tensor) -> torch.Tensor


   .. py:method:: q() -> torch.Tensor


   .. py:method:: from_json(data, dic)
      :classmethod:



.. py:class:: GeneralSymmetricSubstitutionModel(id_: torchtree.typing.ID, data_type: torchtree.evolution.datatype.DataType, mapping: torchtree.core.abstractparameter.AbstractParameter, rates: torchtree.core.abstractparameter.AbstractParameter, frequencies: torchtree.core.abstractparameter.AbstractParameter)

   Bases: :py:obj:`torchtree.evolution.substitution_model.abstract.SymmetricSubstitutionModel`

   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:property:: rates
      :type: torch.Tensor


   .. py:method:: handle_model_changed(model, obj, index)


   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: q() -> torch.Tensor


   .. py:method:: from_json(data, dic)
      :classmethod:



.. py:class:: GeneralNonSymmetricSubstitutionModel(id_: torchtree.typing.ID, data_type: torchtree.evolution.datatype.DataType, mapping: torchtree.core.abstractparameter.AbstractParameter, rates: torchtree.core.abstractparameter.AbstractParameter, frequencies: torchtree.core.abstractparameter.AbstractParameter, normalize: bool)

   Bases: :py:obj:`torchtree.evolution.substitution_model.abstract.NonSymmetricSubstitutionModel`

   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:property:: rates
      :type: torch.Tensor


   .. py:method:: handle_model_changed(model, obj, index)


   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: q() -> torch.Tensor


   .. py:method:: from_json(data, dic)
      :classmethod:



.. py:class:: EmpiricalSubstitutionModel(id_: torchtree.typing.ID, rates: torch.Tensor, frequencies: torch.Tensor)

   Bases: :py:obj:`torchtree.evolution.substitution_model.abstract.SubstitutionModel`

   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:property:: frequencies
      :type: torch.Tensor


   .. py:method:: q() -> torch.Tensor


   .. py:method:: p_t(branch_lengths: torch.Tensor) -> torch.Tensor


   .. py:method:: eigen(Q: torch.Tensor) -> torch.Tensor


   .. py:method:: handle_model_changed(model, obj, index) -> None


   .. py:method:: handle_parameter_changed(variable: torchtree.core.abstractparameter.AbstractParameter, index, event) -> None


   .. py:method:: create_rate_matrix(rates: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor
      :staticmethod:


   .. py:method:: from_json(data, dic)
      :classmethod:



