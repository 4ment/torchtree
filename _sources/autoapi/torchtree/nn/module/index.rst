:py:mod:`torchtree.nn.module`
=============================

.. py:module:: torchtree.nn.module


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.nn.module.Module




.. py:class:: Module(id_: torchtree.typing.ID, module: torch.nn.Module, parameters: torchtree.typing.OrderedDictType[str, torchtree.core.abstractparameter.AbstractParameter])



   Wrapper class for torch.nn.Module.

   :param id_: ID of object.
   :type id_: str or None
   :param torch.nn.Module module: a torch.nn.Module object.
   :param parameters: OrderedDict of :class:`~torchtree.Parameter` keyed by their ID.
   :type parameters: OrderedDict[str,Parameter]

   .. py:property:: module
      :type: torch.nn.Module

      returns torch.nn.Module


   .. py:method:: to(*args, **kwargs) -> None


   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None) -> None


   .. py:method:: cpu() -> None


   .. py:method:: from_json(data, dic)
      :classmethod:

      Create a Module object.

      :param data: json representation of Module object.
      :param dic: dictionary containing additional objects that can be referenced
       in data.

      :return: a :class:`~torchtree.nn.module.Module` object.
      :rtype: Module



