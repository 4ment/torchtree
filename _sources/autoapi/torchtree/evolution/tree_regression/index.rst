:py:mod:`torchtree.evolution.tree_regression`
=============================================

.. py:module:: torchtree.evolution.tree_regression


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   torchtree.evolution.tree_regression.linear_regression



.. py:function:: linear_regression(tree) -> tuple[torch.Tensor, torch.Tensor]

   Calculate rate and root height using linear regression.

   :param tree: Dendropy tree
   :returns:
       - rate - substitution rate
       - root_height - root height


