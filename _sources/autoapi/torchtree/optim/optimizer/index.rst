:py:mod:`torchtree.optim.optimizer`
===================================

.. py:module:: torchtree.optim.optimizer


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.optim.optimizer.Optimizer




.. py:class:: Optimizer(parameters: torchtree.typing.ListParameter, loss: torchtree.core.model.CallableModel, optimizer: torch.optim.Optimizer, iterations: int, **kwargs)

   Bases: :py:obj:`torchtree.core.serializable.JSONSerializable`, :py:obj:`torchtree.core.runnable.Runnable`

   A wrapper for torch.optim.Optimizer objects.

   :param list parameters: list of Parameter
   :param CallableModel loss: loss function
   :param optimizer: a torch.optim.Optimizer object
   :type optimizer: torch.optim.Optimizer
   :param int iterations: number of iterations
   :param kwargs: optionals

   .. py:method:: run() -> None


   .. py:method:: from_json(data: dict[str, any], dic: dict[str, any]) -> Optimizer
      :classmethod:



