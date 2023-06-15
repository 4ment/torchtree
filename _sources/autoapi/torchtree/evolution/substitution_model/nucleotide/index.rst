:py:mod:`torchtree.evolution.substitution_model.nucleotide`
===========================================================

.. py:module:: torchtree.evolution.substitution_model.nucleotide


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.evolution.substitution_model.nucleotide.JC69
   torchtree.evolution.substitution_model.nucleotide.HKY
   torchtree.evolution.substitution_model.nucleotide.GTR




.. py:class:: JC69(id_: torchtree.typing.ID)


   Bases: :py:obj:`torchtree.evolution.substitution_model.abstract.SubstitutionModel`

   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:property:: frequencies
      :type: torch.Tensor


   .. py:method:: p_t(branch_lengths: torch.Tensor) -> torch.Tensor

      Calculate transition probability matrices.

      :param branch_lengths: tensor of branch lengths [B,K]
      :return: tensor of probability matrices [B,K,4,4]


   .. py:method:: q() -> torch.Tensor


   .. py:method:: handle_model_changed(model, obj, index)


   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None) -> None


   .. py:method:: cpu() -> None


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: HKY(id_: torchtree.typing.ID, kappa: torchtree.core.abstractparameter.AbstractParameter, frequencies: torchtree.core.abstractparameter.AbstractParameter)


   Bases: :py:obj:`torchtree.evolution.substitution_model.abstract.SymmetricSubstitutionModel`

   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:property:: kappa
      :type: torch.Tensor


   .. py:method:: handle_model_changed(model, obj, index)


   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: p_t_analytical(branch_lengths: torch.Tensor) -> torch.Tensor
      :abstractmethod:


   .. py:method:: q() -> torch.Tensor


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: GTR(id_: torchtree.typing.ID, rates: torchtree.core.abstractparameter.AbstractParameter, frequencies: torchtree.core.abstractparameter.AbstractParameter)


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

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



