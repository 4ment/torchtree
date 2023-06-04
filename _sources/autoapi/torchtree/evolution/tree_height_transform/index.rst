:py:mod:`torchtree.evolution.tree_height_transform`
===================================================

.. py:module:: torchtree.evolution.tree_height_transform


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.evolution.tree_height_transform.GeneralNodeHeightTransform
   torchtree.evolution.tree_height_transform.DifferenceNodeHeightTransform




.. py:class:: GeneralNodeHeightTransform(tree: TimeTreeModel, cache_size=0)



   Transform from ratios to node heights.

   .. py:attribute:: bijective
      :value: True

      

   .. py:attribute:: sign

      

   .. py:method:: sort_indices()


   .. py:method:: update_bounds() -> None

      Called when topology changes.


   .. py:method:: log_abs_det_jacobian(x, y)

      Computes the log det jacobian `log |dy/dx|` given input and output.



.. py:class:: DifferenceNodeHeightTransform(tree_model: TimeTreeModel, k: float = 0.0, cache_size=0)



   Transform from node height differences to node heights.

   The height :math:`x_i` of node :math:`i` is parameterized as

   .. math::

     x_i = \max(x_{c(i,0)}, x_{c(i,1)}) + y_i

   where :math:`x_c(i,j)` is the height of the jth child of node :math:`i` and
   :math:`y_i \in \mathbb{R}^+`. Function max can be approximated using logsumexp
   in order to propagate the gradient if k > 0.

   .. py:attribute:: bijective
      :value: True

      

   .. py:attribute:: sign

      

   .. py:method:: log_abs_det_jacobian(x, y)

      Computes the log det jacobian `log |dy/dx|` given input and output.



