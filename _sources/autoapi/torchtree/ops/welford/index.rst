torchtree.ops.welford
=====================

.. py:module:: torchtree.ops.welford


Classes
-------

.. autoapisummary::

   torchtree.ops.welford.WelfordVariance


Module Contents
---------------

.. py:class:: WelfordVariance(mean: torch.Tensor, variance: torch.Tensor, samples=0)

   Welford's online method for estimating (co)variance.


   .. py:method:: add_sample(x: torch.Tensor) -> None

      Add sample to calculate mean and variance.

      .. math:
          m_{k} = m_{k-1} + (x - m_{k-1})/k
          v_{k} = v_{k-1} + (x_k - v_{k-1})(x_k - v_{k})



   .. py:method:: remove_sample(x: torch.Tensor)

      Remove sample to calculate mean and variance.

      .. math:
          m_{k-1} = (k m_{k} - x)/(k-1)
          v_{k-1} = v_{k} - (x_k - v_{k-1})(x_k - v_{k})



   .. py:method:: variance() -> torch.Tensor


   .. py:method:: mean() -> torch.Tensor


   .. py:method:: reset() -> None


