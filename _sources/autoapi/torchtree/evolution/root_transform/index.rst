:py:mod:`torchtree.evolution.root_transform`
============================================

.. py:module:: torchtree.evolution.root_transform


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.evolution.root_transform.RootParameter




.. py:class:: RootParameter(id_: torchtree.typing.ID, distance: torchtree.core.parameter.Parameter, rate: torchtree.core.parameter.Parameter, shift: float)


   Bases: :py:obj:`torchtree.core.abstractparameter.AbstractParameter`, :py:obj:`torchtree.core.model.CallableModel`

   This root height parameter is calculated from
    number of substitutions / substitution rate.

   :param id_: ID of object
   :type id_: str or None
   :param Parameter distance: number of substitution parameter
   :param Parameter rate: rate parameter
   :param float shift: shift root height by this amount. Used by serially sampled trees

   .. py:property:: tensor
      :type: torch.Tensor

      The tensor.

      :getter: Returns the tensor.
      :setter: Sets the tensor.
      :rtype: Tensor

   .. py:method:: parameters() -> list[torchtree.core.parameter.Parameter]

      Returns parameters of instance Parameter.


   .. py:method:: transform() -> torch.Tensor

      Return root height.


   .. py:method:: handle_parameter_changed(variable, index, event) -> None


   .. py:method:: handle_model_changed(model, obj, index) -> None


   .. py:method:: add_parameter_listener(listener) -> None


   .. py:method:: fire_parameter_changed(index=None, event=None) -> None


   .. py:method:: from_json(data, dic) -> RootParameter
      :classmethod:

      Create a RootParameter object.

      :param data: json representation of RootParameter object.
      :type data: dict[str,Any]
      :param dic: dictionary containing additional objects that can be referenced
       in data.
      :type dic: dict[str,Any]

      :return: a :class:`~torchtree.evolution.root_transform.RootParameter` object.
      :rtype: RootParameter



