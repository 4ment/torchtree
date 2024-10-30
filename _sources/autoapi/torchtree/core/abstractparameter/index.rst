torchtree.core.abstractparameter
================================

.. py:module:: torchtree.core.abstractparameter

.. autoapi-nested-parse::

   Abstract parameter module.



Classes
-------

.. autoapisummary::

   torchtree.core.abstractparameter.Device
   torchtree.core.abstractparameter.AbstractParameter


Module Contents
---------------

.. py:class:: Device

   Bases: :py:obj:`abc.ABC`


   Interface for classes needing to allocate a Tensor on a device.


   .. py:property:: device
      :type: torch.device

      :abstractmethod:


      Returns the torch.device where the Tensor is.

      :rtype: torch.device


   .. py:method:: to(*args, **kwargs) -> None
      :abstractmethod:


      Performs Tensor dtype and/or device conversion.



   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None) -> None
      :abstractmethod:


      Moves the tensor object in CUDA memory.



   .. py:method:: cpu() -> None
      :abstractmethod:


      Moves the tensor object in CPU memory.



.. py:class:: AbstractParameter(id_: Optional[str])

   Bases: :py:obj:`torchtree.core.identifiable.Identifiable`, :py:obj:`Device`, :py:obj:`abc.ABC`


   Abstract base class for parameters.


   .. py:property:: tensor
      :type: torch.Tensor

      :abstractmethod:


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


   .. py:method:: dim() -> int

      Returns the dimension of the tensor.

      :rtype: int



   .. py:method:: parameters() -> List[AbstractParameter]


   .. py:property:: device
      :type: torch.device


      Returns the torch.device where the Tensor is.

      :rtype: torch.device


   .. py:method:: add_parameter_listener(listener) -> None
      :abstractmethod:



   .. py:method:: fire_parameter_changed(index=None, event=None) -> None
      :abstractmethod:



