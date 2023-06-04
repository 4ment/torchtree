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



   A wrapper for :class:`~torch.optim.lr_scheduler` objects.

   :param scheduler: a :class:`~torch.optim.lr_scheduler`

   .. py:method:: step(*args) -> None


   .. py:method:: from_json(data: dict[str, any], dic: dict[str, any], **kwargs) -> Scheduler
      :classmethod:



