torchtree.evolution.substitution_model.codon
============================================

.. py:module:: torchtree.evolution.substitution_model.codon


Classes
-------

.. autoapisummary::

   torchtree.evolution.substitution_model.codon.MG94


Module Contents
---------------

.. py:class:: MG94(id_: torchtree.typing.ID, data_type: torchtree.evolution.datatype.CodonDataType, alpha: torchtree.core.abstractparameter.AbstractParameter, beta: torchtree.core.abstractparameter.AbstractParameter, kappa: torchtree.core.abstractparameter.AbstractParameter, frequencies: torchtree.core.abstractparameter.AbstractParameter)

   Bases: :py:obj:`torchtree.evolution.substitution_model.abstract.SymmetricSubstitutionModel`


   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.


   .. py:method:: q() -> torch.Tensor


   .. py:property:: rates
      :type: Union[torch.Tensor, list[torch.Tensor]]



   .. py:method:: handle_model_changed(model, obj, index) -> None


   .. py:method:: handle_parameter_changed(variable: torchtree.core.abstractparameter.AbstractParameter, index, event) -> None


   .. py:method:: from_json(data, dic)
      :classmethod:


      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



