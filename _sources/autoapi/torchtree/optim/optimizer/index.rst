torchtree.optim.optimizer
=========================

.. py:module:: torchtree.optim.optimizer


Classes
-------

.. autoapisummary::

   torchtree.optim.optimizer.Optimizer


Module Contents
---------------

.. py:class:: Optimizer(id_: torchtree.typing.ID, parameters: torchtree.typing.ListParameter, loss: torchtree.core.model.CallableModel, optimizer: torch.optim.Optimizer, iterations: int, **kwargs)

   Bases: :py:obj:`torchtree.core.identifiable.Identifiable`, :py:obj:`torchtree.core.runnable.Runnable`


   A wrapper for torch.optim.Optimizer objects.

   :param list parameters: list of Parameter
   :param CallableModel loss: loss function
   :param optimizer: a torch.optim.Optimizer object
   :type optimizer: torch.optim.Optimizer
   :param int iterations: number of iterations
   :param kwargs: optionals


   .. py:method:: run() -> None


   .. py:method:: state_dict() -> dict[str, Any]


   .. py:method:: load_state_dict(state_dict: dict[str, Any]) -> None


   .. py:method:: save_full_state(checkpoint, safely=True, overwrite=False) -> None


   .. py:method:: from_json(data: dict[str, Any], dic: dict[str, Any]) -> Optimizer
      :classmethod:


      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



