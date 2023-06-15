:py:mod:`torchtree.optim.lr_scheduler`
======================================

.. py:module:: torchtree.optim.lr_scheduler


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.optim.lr_scheduler.Scheduler




.. py:class:: Scheduler(scheduler: torch.optim.lr_scheduler._LRScheduler)


   Bases: :py:obj:`torchtree.core.serializable.JSONSerializable`

   A wrapper for :class:`~torch.optim.lr_scheduler` objects.

   :param scheduler: a :class:`~torch.optim.lr_scheduler`

   .. py:method:: step(*args) -> None


   .. py:method:: from_json(data: dict[str, any], dic: dict[str, any], **kwargs) -> Scheduler
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



