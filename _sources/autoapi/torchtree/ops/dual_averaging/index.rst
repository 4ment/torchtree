torchtree.ops.dual_averaging
============================

.. py:module:: torchtree.ops.dual_averaging


Classes
-------

.. autoapisummary::

   torchtree.ops.dual_averaging.DualAveraging


Module Contents
---------------

.. py:class:: DualAveraging(mu=0.5, gamma=0.05, kappa=0.75, t0=10)

   Dual averaging Nesterov.

   Code adapted from: https://github.com/stan-dev/stan


   .. py:method:: restart() -> None


   .. py:method:: step(statistic) -> None


