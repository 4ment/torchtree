:py:mod:`torchtree.core.parameter`
==================================

.. py:module:: torchtree.core.parameter

.. autoapi-nested-parse::

   Implementation of Parameter classes.



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


   Bases: :py:obj:`torchtree.core.abstractparameter.AbstractParameter`

   Parameter class.

   :param id_: identifier of Parameter object.
   :type id_: str or None
   :param Tensor tensor: Tensor object.

   .. py:property:: tensor
      :type: torch.Tensor

      The tensor.

      :getter: Returns the tensor.
      :setter: Sets the tensor.
      :rtype: Tensor

   .. py:property:: requires_grad
      :type: bool

      Is True if gradients need to be computed for this Tensor, False otherwise.

      :getter: Returns the flag.
      :setter: Sets the flag.
      :rtype: bool

   .. py:property:: grad_fn

      The grad_fn property of the tensor.

      :rtype: torch.autograd.graph.node

   .. py:property:: grad
      :type: torch.Tensor

      The grad property of the Tensor.

      :rtype: Tensor

   .. py:method:: detach()


   .. py:method:: copy_(tensor)


   .. py:method:: size() -> torch.Size

      Returns the size of the tensor.

      :rtype: Size


   .. py:method:: add_parameter_listener(listener) -> None


   .. py:method:: fire_parameter_changed(index=None, event=None) -> None


   .. py:method:: clone() -> Parameter

      Return a clone of the Parameter.

      it is not cloning listeners and the clone's id is None


   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None) -> None

      Moves the tensor object in CUDA memory.


   .. py:method:: cpu() -> None

      Moves the tensor object in CPU memory.


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


   .. py:method:: from_json(data: dict[str, Any], dic: dict[str, torchtree.core.identifiable.Identifiable]) -> Parameter
      :classmethod:

      Creates a Parameter object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a parameter object.
      :param dict[str, Identifiable] dic: dictionary containing torchtree objects
          keyed by their ID.

      **JSON attributes**:

       Only one of :attr:`tensor`, :attr:`full_like` :attr:`full`,
       :attr:`zeros_like`, :attr:`zeros`, :attr:`ones_like`, :attr:`ones`,
       :attr:`eye_like`, :attr:`eye`, :attr:`arange` can be specified.

       - tensor (list): list of values.
       - full_like (AbstractParameter): parameter used to determine the size of
         the tensor.

         - value (float or int or bool): the number to fill the tensor with.
       - full (int or list): size of the tensor.

         - value (float or int or bool): the number to fill the tensor with.
       - ones_like (AbstractParameter): parameter used to determine the size of
         the tensor filled with the scalar value 1.
       - ones (int or list): size of the tensor.
       - zeros_like (AbstractParameter): parameter used to determine the size of
         the tensor filled with the scalar value 0.
       - zeros (int or list): size of the tensor.
       - eye_like (AbstractParameter): parameter used to create a 2-D tensor with
         ones on the diagonal and zeros elsewhere.
       - eye (int or list): size of the 2D tensor with ones on the diagonal and
         zeros elsewhere. The list can only contain 2 integers.
       - arange (int or list): emulate torch.arange. If a int is provided it is
         equiavalent to torch.arange(x). If a list is provided it is equivalient to
         torch.arange(x[0], x[1], x[2]). The list can be of size 2 or 3.

       Optional:
        - dtype (str): the desired data type of returned tensor.
          Default: if None, infers data type from data.
        - device (str):  the device of the constructed tensor. If None and data
          is a tensor then the device of data is used. If None and data is not a
          tensor then the result tensor is constructed on the CPU.
        - requires_grad (bool): If autograd should record operations on the returned
          tensor. Default: False.
        - nn (bool): If the tensor should wrapped in a torch.nn.Parameter object.

      :example:
      >>> p_dic = {"id": "parameter", "type": "Parameter", "tensor": [1., 2., 3.]}
      >>> parameter = Parameter.from_json(p_dic, {})
      >>> isinstance(parameter, Parameter)
      True
      >>> parameter.tensor
      tensor([1., 2., 3.])
      >>> ones_dic = {"id": "parameter", "type": "Parameter", "ones_like": p_dic}
      >>> ones = Parameter.from_json(ones_dic, {})
      >>> all(ones.tensor == torch.ones(3))
      True

      .. note::
          The specification of the tensor loosely follows the way Tensors
          (full, ones, eye, ...) are constructed:
          https://pytorch.org/docs/stable/torch.html



.. py:class:: TransformedParameter(id_: Optional[str], x: Union[list[torchtree.core.abstractparameter.AbstractParameter], torchtree.core.abstractparameter.AbstractParameter], transform: torch.distributions.Transform)


   Bases: :py:obj:`torchtree.core.abstractparameter.AbstractParameter`, :py:obj:`torchtree.core.parametric.Parametric`, :py:obj:`collections.abc.Callable`

   Class wrapping an AbstractParameter and a torch Transform object.

   The tensor property of this object returns the wrapped parameter tensor
   transformed with the wrapped transform.

   This class is callable and it returns the log determinant jacobians of the
   invertible transformation.

   :param id_: object identifier.
   :type id_: str or None
   :param x: parameter to transform.
   :type x: Union[list[AbstractParameter], AbstractParameter]
   :param transform: torch transform object.
   :type transform: torch.distributions.Transform

   .. py:property:: tensor
      :type: torch.Tensor

      The tensor.

      :getter: Returns the tensor.
      :setter: Sets the tensor.
      :rtype: Tensor

   .. py:property:: requires_grad
      :type: bool

      Is True if gradients need to be computed for this Tensor, False otherwise.

      :getter: Returns the flag.
      :setter: Sets the flag.
      :rtype: bool

   .. py:property:: shape
      :type: torch.Size

      The shape of the tensor.

      :rtype: Size

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

      Performs Tensor dtype and/or device conversion.


   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None)

      Moves the tensor object in CUDA memory.


   .. py:method:: cpu()

      Moves the tensor object in CPU memory.


   .. py:method:: from_json(data: dict[str, Any], dic: dict[str, torchtree.core.identifiable.Identifiable]) -> TransformedParameter
      :classmethod:

      Creates a TransformedParameter object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a transformed
          parameter object.
      :param dict[str, Identifiable] dic: dictionary containing torchtree objects
          keyed by their ID.

      **JSON attributes**:

       Mandatory:
        - id (str): identidifer of object.
        - x (AbstractParameter): parameter to transform.
        - transform (str): torch transform class name, including package
          and module names.

       Optional:
        - parameters (dic): parameter of torch transform.

      :example:
      >>> tensor = torch.tensor([1.,2.])
      >>> p_dic = {"id": "parameter", "type": "Parameter", "tensor": tensor.tolist()}
      >>> t_dic =  {"id": "t", "type": "TransformedParameter", "x": p_dic,
      ... "transform": "torch.distributions.ExpTransform"}
      >>> transformed = TransformedParameter.from_json(t_dic, {})
      >>> isinstance(transformed, TransformedParameter)
      True
      >>> exp_transform = torch.distributions.ExpTransform()
      >>> tensor2 = exp_transform(tensor)
      >>> all(transformed.tensor == tensor2)
      True
      >>> all(transformed() == exp_transform.log_abs_det_jacobian(tensor, tensor2))
      True



.. py:class:: ViewParameter(id_: Optional[str], parameter: Parameter, indices: Union[int, slice, torch.Tensor])


   Bases: :py:obj:`torchtree.core.abstractparameter.AbstractParameter`, :py:obj:`torchtree.core.parametric.ParameterListener`

   Class representing a view of another parameter.

   :param id_: ID of object.
   :type id_: str or None
   :param Parameter parameter: parameter that ViewParameter wrap.
   :param indices: indices used on parameter

   .. py:property:: tensor
      :type: torch.Tensor

      The tensor.

      :getter: Returns the tensor.
      :setter: Sets the tensor.
      :rtype: Tensor

   .. py:property:: shape
      :type: torch.Size

      The shape of the tensor.

      :rtype: Size

   .. py:property:: dtype
      :type: torch.dtype

      The dtype of the tensor.

      :rtype: torch.dtype

   .. py:property:: requires_grad
      :type: bool

      Is True if gradients need to be computed for this Tensor, False otherwise.

      :getter: Returns the flag.
      :setter: Sets the flag.
      :rtype: bool

   .. py:method:: assign(parameter)


   .. py:method:: add_parameter_listener(listener) -> None


   .. py:method:: fire_parameter_changed(index=None, event=None) -> None


   .. py:method:: clone() -> ViewParameter

      Return a clone of the Parameter.

      it is not cloning listeners and the clone's id is None


   .. py:method:: handle_parameter_changed(variable, index, event) -> None


   .. py:method:: to(*args, **kwargs) -> None

      Performs Tensor dtype and/or device conversion.


   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None) -> None

      Moves the tensor object in CUDA memory.


   .. py:method:: cpu() -> None

      Moves the tensor object in CPU memory.


   .. py:method:: json_factory(id_: str, x, indices)
      :staticmethod:


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: CatParameter(id_: Optional[str], parameters: Union[list[Parameter], tuple[Parameter, Ellipsis]], dim: Optional[int] = 0)


   Bases: :py:obj:`torchtree.core.abstractparameter.AbstractParameter`, :py:obj:`torchtree.core.parametric.ParameterListener`

   Class for concatenating parameters.

   :param id_: ID of object
   :param parameters: list or tuple of parameters
   :param dim: dimension for concatenation

   .. py:property:: tensor
      :type: torch.Tensor

      The tensor.

      :getter: Returns the tensor.
      :setter: Sets the tensor.
      :rtype: Tensor

   .. py:property:: requires_grad
      :type: bool

      Is True if gradients need to be computed for this Tensor, False otherwise.

      :getter: Returns the flag.
      :setter: Sets the flag.
      :rtype: bool

   .. py:property:: device
      :type: torch.device

      Returns the torch.device where the Tensor is.

      :rtype: torch.device

   .. py:method:: update()


   .. py:method:: to(*args, **kwargs) -> None

      Performs Tensor dtype and/or device conversion.


   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None) -> None

      Moves the tensor object in CUDA memory.


   .. py:method:: cpu() -> None

      Moves the tensor object in CPU memory.


   .. py:method:: add_parameter_listener(listener) -> None


   .. py:method:: fire_parameter_changed(index=None, event=None) -> None


   .. py:method:: handle_model_changed(variable, index, event) -> None


   .. py:method:: handle_parameter_changed(variable, index, event) -> None


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: ModuleParameter(id_: Optional[str], module)


   Bases: :py:obj:`torchtree.core.abstractparameter.AbstractParameter`, :py:obj:`torchtree.core.parametric.Parametric`

   Abstract base class for parameters.

   .. py:property:: tensor
      :type: torch.Tensor

      The tensor.

      :getter: Returns the tensor.
      :setter: Sets the tensor.
      :rtype: Tensor

   .. py:property:: requires_grad
      :type: bool

      Is True if gradients need to be computed for this Tensor, False otherwise.

      :getter: Returns the flag.
      :setter: Sets the flag.
      :rtype: bool

   .. py:property:: shape
      :type: torch.Size

      The shape of the tensor.

      :rtype: Size

   .. py:property:: sample_shape
      :type: torch.Size


   .. py:method:: parameters() -> list[torchtree.core.abstractparameter.AbstractParameter]

      Returns parameters of instance Parameter.


   .. py:method:: handle_parameter_changed(variable, index, event) -> None


   .. py:method:: handle_model_changed(model, obj, index) -> None


   .. py:method:: add_parameter_listener(listener) -> None


   .. py:method:: fire_parameter_changed(index=None, event=None) -> None


   .. py:method:: to(*args, **kwargs) -> None

      Performs Tensor dtype and/or device conversion.


   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None)

      Moves the tensor object in CUDA memory.


   .. py:method:: cpu()

      Moves the tensor object in CPU memory.


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



