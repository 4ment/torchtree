:py:mod:`torchtree.evolution.substitution_model.abstract`
=========================================================

.. py:module:: torchtree.evolution.substitution_model.abstract


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.evolution.substitution_model.abstract.SubstitutionModel
   torchtree.evolution.substitution_model.abstract.AbstractSubstitutionModel
   torchtree.evolution.substitution_model.abstract.SymmetricSubstitutionModel
   torchtree.evolution.substitution_model.abstract.NonSymmetricSubstitutionModel




.. py:class:: SubstitutionModel(id_: torchtree.typing.ID)



   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:property:: frequencies
      :type: torch.Tensor
      :abstractmethod:


   .. py:method:: p_t(branch_lengths: torch.Tensor) -> torch.Tensor
      :abstractmethod:


   .. py:method:: q() -> torch.Tensor
      :abstractmethod:



.. py:class:: AbstractSubstitutionModel(id_: torchtree.typing.ID, frequencies: torchtree.core.abstractparameter.AbstractParameter)



   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:property:: frequencies
      :type: torch.Tensor


   .. py:method:: norm(Q) -> torch.Tensor



.. py:class:: SymmetricSubstitutionModel(id_: torchtree.typing.ID, frequencies: torchtree.core.abstractparameter.AbstractParameter)



   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:method:: p_t(branch_lengths: torch.Tensor) -> torch.Tensor


   .. py:method:: eigen(Q: torch.Tensor) -> torch.Tensor



.. py:class:: NonSymmetricSubstitutionModel(id_: torchtree.typing.ID, frequencies: torchtree.core.abstractparameter.AbstractParameter)



   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:method:: p_t(branch_lengths: torch.Tensor) -> torch.Tensor


   .. py:method:: eigen(Q: torch.Tensor) -> torch.Tensor



