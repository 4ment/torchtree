:py:mod:`torchtree.core.model`
==============================

.. py:module:: torchtree.core.model

.. autoapi-nested-parse::

   Parametric models.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.core.model.Model
   torchtree.core.model.CallableModel




.. py:class:: Model(id_: Optional[str])


   Bases: :py:obj:`torchtree.core.parametric.Parametric`, :py:obj:`torchtree.core.identifiable.Identifiable`, :py:obj:`torchtree.core.parametric.ModelListener`, :py:obj:`torchtree.core.parametric.ParameterListener`

   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:property:: tag
      :type: Optional[str]


   .. py:property:: sample_shape
      :type: torch.Size

      Returns sample shape.

   .. py:method:: add_model_listener(listener: torchtree.core.parametric.ModelListener) -> None


   .. py:method:: remove_model_listener(listener: torchtree.core.parametric.ModelListener) -> None


   .. py:method:: add_parameter_listener(listener: torchtree.core.parametric.ParameterListener) -> None


   .. py:method:: remove_parameter_listener(listener: torchtree.core.parametric.ParameterListener) -> None


   .. py:method:: fire_model_changed(obj=None, index=None) -> None


   .. py:method:: to(*args, **kwargs) -> None

      Performs Tensor dtype and/or device conversion using torch.to.


   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None) -> None

      Move tensors to CUDA using torch.cuda.


   .. py:method:: cpu() -> None

      Move tensors to CPU memory using ~torch.cpu.


   .. py:method:: models()

      Returns sub-models.



.. py:class:: CallableModel(id_: Optional[str])


   Bases: :py:obj:`Model`, :py:obj:`collections.abc.Callable`

   Classes inheriting from :class:`Model` and
   :class:`collections.abc.Callable`.

   CallableModel are Callable and the returned value is cached in case
   we need to use this value multiple times without the need to
   recompute it.

   .. py:method:: handle_parameter_changed(variable: torchtree.core.abstractparameter.AbstractParameter, index, event) -> None


   .. py:method:: handle_model_changed(model, obj, index) -> None



