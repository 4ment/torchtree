torchtree.nn.module
===================

.. py:module:: torchtree.nn.module


Classes
-------

.. autoapisummary::

   torchtree.nn.module.Module


Module Contents
---------------

.. py:class:: Module(id_: torchtree.typing.ID, module: torch.nn.Module, parameters: torchtree.typing.OrderedDictType[str, torchtree.core.abstractparameter.AbstractParameter])

   Bases: :py:obj:`torchtree.core.model.CallableModel`


   Wrapper class for torch.nn.Module.

   :param id_: ID of object.
   :type id_: str or None
   :param torch.nn.Module module: a torch.nn.Module object.
   :param parameters: OrderedDict of :class:`~torchtree.Parameter` keyed by their ID.
   :type parameters: OrderedDict[str,Parameter]


   .. py:attribute:: x


   .. py:property:: module
      :type: torch.nn.Module


      returns torch.nn.Module


   .. py:method:: to(*args, **kwargs) -> None

      Performs Tensor dtype and/or device conversion using torch.to.



   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None) -> None

      Move tensors to CUDA using torch.cuda.



   .. py:method:: cpu() -> None

      Move tensors to CPU memory using ~torch.cpu.



   .. py:method:: from_json(data, dic)
      :classmethod:


      Create a Module object.

      :param data: json representation of Module object.
      :param dic: dictionary containing additional objects that can be referenced
       in data.

      :return: a :class:`~torchtree.nn.module.Module` object.
      :rtype: Module



