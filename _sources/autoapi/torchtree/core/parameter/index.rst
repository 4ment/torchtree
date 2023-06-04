:py:mod:`torchtree.core.parameter`
==================================

.. py:module:: torchtree.core.parameter


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.core.parameter.Parameter
   torchtree.core.parameter.TransformedParameter
   torchtree.core.parameter.ViewParameter
   torchtree.core.parameter.CatParameter
   torchtree.core.parameter.ModuleParameter




.. py:class:: Parameter(id_: Optional[str], tensor: torch.Tensor)



   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:property:: tensor
      :type: torch.Tensor


   .. py:property:: requires_grad
      :type: bool


   .. py:property:: grad_fn


   .. py:property:: grad
      :type: bool


   .. py:method:: detach()


   .. py:method:: copy_(tensor)


   .. py:method:: size()


   .. py:method:: add_parameter_listener(listener) -> None


   .. py:method:: fire_parameter_changed(index=None, event=None) -> None


   .. py:method:: clone() -> Parameter

      Return a clone of the Parameter.

      it is not cloning listeners and the clone's id is None


   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None) -> None


   .. py:method:: cpu() -> None


   .. py:method:: to(device: Optional[Union[int, torch.device]] = None, dtype: Optional[Union[torch.dtype, str]] = None) -> None
               to(dtype: Union[torch.dtype, str] = None) -> None

      Performs Tensor dtype and/or device conversion.

      A torch.dtype and torch.device are inferred from the arguments
      of self.to(*args, **kwargs)

      This can be called as

      .. function:: to(device=None, dtype=None)

      .. function:: to(dtype)

      .. function:: to(device)


   .. py:method:: json_factory(id_: str, **kwargs)
      :staticmethod:


   .. py:method:: from_json(data, dic)
      :classmethod:



.. py:class:: TransformedParameter(id_: Optional[str], x: Union[list[torchtree.core.abstractparameter.AbstractParameter], torchtree.core.abstractparameter.AbstractParameter], transform: torch.distributions.Transform)



   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:property:: tensor
      :type: torch.Tensor


   .. py:property:: requires_grad
      :type: bool


   .. py:property:: shape
      :type: torch.Size


   .. py:property:: sample_shape
      :type: torch.Size


   .. py:method:: parameters() -> list[torchtree.core.abstractparameter.AbstractParameter]

      Returns parameters of instance Parameter.


   .. py:method:: apply_transform() -> None


   .. py:method:: handle_parameter_changed(variable, index, event) -> None


   .. py:method:: handle_model_changed(model, obj, index) -> None


   .. py:method:: add_parameter_listener(listener) -> None


   .. py:method:: fire_parameter_changed(index=None, event=None) -> None


   .. py:method:: to(*args, **kwargs) -> None


   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None)


   .. py:method:: cpu()


   .. py:method:: from_json(data, dic)
      :classmethod:



.. py:class:: ViewParameter(id_: Optional[str], parameter: Parameter, indices: Union[int, slice, torch.Tensor])



   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:property:: tensor
      :type: torch.Tensor


   .. py:property:: shape
      :type: torch.Size


   .. py:property:: dtype
      :type: torch.dtype


   .. py:property:: requires_grad
      :type: bool


   .. py:method:: assign(parameter)


   .. py:method:: add_parameter_listener(listener) -> None


   .. py:method:: fire_parameter_changed(index=None, event=None) -> None


   .. py:method:: clone() -> ViewParameter

      Return a clone of the Parameter.

      it is not cloning listeners and the clone's id is None


   .. py:method:: handle_parameter_changed(variable, index, event) -> None


   .. py:method:: to(*args, **kwargs) -> None


   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None) -> None


   .. py:method:: cpu() -> None


   .. py:method:: json_factory(id_: str, x, indices)
      :staticmethod:


   .. py:method:: from_json(data, dic)
      :classmethod:



.. py:class:: CatParameter(id_: Optional[str], parameters: Union[list[Parameter], tuple[Parameter, Ellipsis]], dim: Optional[int] = 0)



   Class for concatenating parameters.

   :param id_: ID of object
   :param parameters: list or tuple of parameters
   :param dim: dimension for concatenation

   .. py:property:: tensor
      :type: torch.Tensor


   .. py:property:: requires_grad
      :type: bool


   .. py:property:: device
      :type: torch.device


   .. py:method:: update()


   .. py:method:: to(*args, **kwargs) -> None


   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None) -> None


   .. py:method:: cpu() -> None


   .. py:method:: add_parameter_listener(listener) -> None


   .. py:method:: fire_parameter_changed(index=None, event=None) -> None


   .. py:method:: handle_model_changed(variable, index, event) -> None


   .. py:method:: handle_parameter_changed(variable, index, event) -> None


   .. py:method:: from_json(data, dic)
      :classmethod:



.. py:class:: ModuleParameter(id_: Optional[str], module)



   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:property:: tensor
      :type: torch.Tensor


   .. py:property:: requires_grad
      :type: bool


   .. py:property:: shape
      :type: torch.Size


   .. py:property:: sample_shape
      :type: torch.Size


   .. py:method:: parameters() -> list[torchtree.core.abstractparameter.AbstractParameter]

      Returns parameters of instance Parameter.


   .. py:method:: handle_parameter_changed(variable, index, event) -> None


   .. py:method:: handle_model_changed(model, obj, index) -> None


   .. py:method:: add_parameter_listener(listener) -> None


   .. py:method:: fire_parameter_changed(index=None, event=None) -> None


   .. py:method:: to(*args, **kwargs) -> None


   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None)


   .. py:method:: cpu()


   .. py:method:: from_json(data, dic)
      :classmethod:



