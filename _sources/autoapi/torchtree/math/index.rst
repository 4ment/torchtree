:py:mod:`torchtree.math`
========================

.. py:module:: torchtree.math


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   torchtree.math.soft_sort
   torchtree.math.soft_max
   torchtree.math.soft_searchsorted



.. py:function:: soft_sort(s: torch.Tensor, tau: float) -> torch.Tensor

   Continuous relaxation for the argsort operator [#Prillo2020]_

   :param Tensor s: input tensor
   :param float tau: temperature
   :return: permutation matrix
   :rtype: Tensor

   .. [#Prillo2020] Prillo & Eisenschlos. SoftSort: A Continuous Relaxation for
   the argsort Operator. 2020.


.. py:function:: soft_max(input: torch.Tensor, k: float, dim, keepdim=False)


.. py:function:: soft_searchsorted(sorted_sequence: torch.Tensor, values: torch.Tensor, tau)

   Continuous relaxation for the torch.serachsorted function

   :param Tensor sorted_sequence: N-D or 1-D tensor, containing monotonically
       increasing sequence on the innermost dimension
   :param Tensor values: N-D tensor or a Scalar containing the search value(s)
   :param float tau: temperature
   :return: selection matrix
   :rtype: Tensor

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


