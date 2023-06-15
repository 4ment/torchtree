:py:mod:`torchtree.evolution.attribute_pattern`
===============================================

.. py:module:: torchtree.evolution.attribute_pattern


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.evolution.attribute_pattern.AttributePattern




.. py:class:: AttributePattern(id_: Optional[str], taxa: torchtree.evolution.taxa.Taxa, data_type: torchtree.evolution.datatype.DataType, attribute: str)


   Bases: :py:obj:`torchtree.core.model.Model`

   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:method:: compute_tips_states()


   .. py:method:: compute_tips_partials(use_ambiguities=False)


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



