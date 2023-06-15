:py:mod:`torchtree.ops.welford`
===============================

.. py:module:: torchtree.ops.welford


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.ops.welford.WelfordVariance




.. py:class:: WelfordVariance(mean: torch.Tensor, variance: torch.Tensor, samples=0)


   Welford's online method for estimating (co)variance.

   .. py:method:: add_sample(x: torch.Tensor) -> None


   .. py:method:: variance() -> torch.Tensor


   .. py:method:: mean() -> torch.Tensor


   .. py:method:: reset() -> None



