:py:mod:`torchtree.core.abstractparameter`
==========================================

.. py:module:: torchtree.core.abstractparameter


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.core.abstractparameter.Device
   torchtree.core.abstractparameter.AbstractParameter




.. py:class:: Device

   Bases: :py:obj:`abc.ABC`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:property:: device
      :type: torch.device
      :abstractmethod:


   .. py:method:: to(*args, **kwargs) -> None
      :abstractmethod:


   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None) -> None
      :abstractmethod:


   .. py:method:: cpu() -> None
      :abstractmethod:



.. py:class:: AbstractParameter(id_: Optional[str])

   Bases: :py:obj:`torchtree.core.identifiable.Identifiable`, :py:obj:`Device`, :py:obj:`abc.ABC`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:property:: tensor
      :type: torch.Tensor
      :abstractmethod:


   .. py:property:: shape
      :type: torch.Size


   .. py:property:: dtype
      :type: torch.dtype


   .. py:property:: requires_grad
      :type: bool


   .. py:property:: device
      :type: torch.device


   .. py:method:: dim() -> int


   .. py:method:: parameters() -> List[AbstractParameter]


   .. py:method:: add_parameter_listener(listener) -> None
      :abstractmethod:


   .. py:method:: fire_parameter_changed(index=None, event=None) -> None
      :abstractmethod:



