torchtree.evolution.attribute_pattern
=====================================

.. py:module:: torchtree.evolution.attribute_pattern


Classes
-------

.. autoapisummary::

   torchtree.evolution.attribute_pattern.AttributePattern


Module Contents
---------------

.. py:class:: AttributePattern(id_: Optional[str], taxa: torchtree.evolution.taxa.Taxa, data_type: torchtree.evolution.datatype.DataType, attribute: str)

   Bases: :py:obj:`torchtree.core.model.Model`


   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.


   .. py:attribute:: taxa


   .. py:attribute:: data_type


   .. py:attribute:: attribute


   .. py:method:: compute_tips_states()


   .. py:method:: compute_tips_partials(use_ambiguities=False)


   .. py:method:: handle_model_changed(model, obj, index)


   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None) -> None

      Move tensors to CUDA using torch.cuda.



   .. py:method:: cpu() -> None

      Move tensors to CPU memory using ~torch.cpu.



   .. py:method:: from_json(data, dic)
      :classmethod:


      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



