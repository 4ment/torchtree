torchtree.evolution.tree_regression
===================================

.. py:module:: torchtree.evolution.tree_regression


Functions
---------

.. autoapisummary::

   torchtree.evolution.tree_regression.linear_regression


Module Contents
---------------

.. py:function:: linear_regression(tree) -> tuple[torch.Tensor, torch.Tensor]

   Calculate rate and root height using linear regression.

   :param tree: Dendropy tree
   :returns:
       - rate - substitution rate
       - root_height - root height


