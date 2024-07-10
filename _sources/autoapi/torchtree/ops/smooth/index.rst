torchtree.ops.smooth
====================

.. py:module:: torchtree.ops.smooth

.. autoapi-nested-parse::

   Continuous relaxation of operators.



Functions
---------

.. autoapisummary::

   torchtree.ops.smooth.soft_sort
   torchtree.ops.smooth.smooth_max
   torchtree.ops.smooth.soft_searchsorted


Module Contents
---------------

.. py:function:: soft_sort(tensor: torch.Tensor, tau: float) -> torch.Tensor

   Continuous relaxation for the argsort operator.

   This method was proposed by :footcite:t:`prillo2020softsort`.

   :param ~torch.Tensor tensor: input tensor
   :param float tau: temperature
   :return: permutation matrix
   :rtype: ~torch.Tensor

   .. footbibliography::


.. py:function:: smooth_max(tensor: torch.Tensor, k: float, dim, keepdim=False) -> torch.Tensor

   Smooth maximum function using `torch.logsumexp <https://pytorch.org/docs/    stable/generated/torch.logsumexp.html#torch.logsumexp>`_.

   :param ~torch.Tensor tensor: input tensor
   :param float k: temperature.
   :param int or tuple of ints dim: the dimension or dimensions to reduce.
   :param bool keepdim: whether the output tensor has dim retained or not.
   :return: smoothed maximum
   :rtype: ~torch.Tensor


.. py:function:: soft_searchsorted(sorted_sequence: torch.Tensor, values: torch.Tensor, tau: float) -> torch.Tensor

   Continuous relaxation of `torch.searchsorted <https://pytorch.org/docs/stable/    generated/torch.searchsorted.html#torch.searchsorted>`_ operator.

   :param Tensor sorted_sequence: N-D or 1-D tensor, containing monotonically
       increasing sequence on the innermost dimension.
   :param ~torch.Tensor values: N-D tensor or a Scalar containing the search value(s).
   :param float tau: temperature.
   :return: selection matrix
   :rtype: ~torch.Tensor

   :example:
   >>> sorted_sequence = torch.tensor((-1., 10, 100.))
   >>> soft_searchsorted(sorted_sequence, torch.tensor(11.), 0.0001)
   tensor([[0., 0., 1., 0.]])
   >>> values = torch.tensor((0., 5000., 30.))
   >>> soft_selection = soft_searchsorted(sorted_sequence, values, 0.0001)
   >>> soft_selection
   tensor([[0., 1., 0., 0.],
           [0., 0., 0., 1.],
           [0., 0., 1., 0.]])
   >>> indices = torch.searchsorted(sorted_sequence, values)
   >>> indices
   tensor([1, 3, 2])
   >>> torch.argmax(soft_selection, -1) == indices
   tensor([True, True, True])


