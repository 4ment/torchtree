torchtree.distributions.log_normal
==================================

.. py:module:: torchtree.distributions.log_normal

.. autoapi-nested-parse::

   Lognormal distribution parametrized by mean and scale.



Classes
-------

.. autoapisummary::

   torchtree.distributions.log_normal.LogNormal


Module Contents
---------------

.. py:class:: LogNormal(mean: Union[torch.Tensor, float], scale: Union[torch.Tensor, float, None] = None, stdev: Union[torch.Tensor, float, None] = None, validate_args=None)

   Bases: :py:obj:`torch.distributions.LogNormal`


   Lognormal distribution parametrized by mean and scale.

   Creates a lognormal distribution parameterized by :attr:`mean` and
   :attr:`scale`.

   :param mean: mean of the distribution
   :param scale: scale (sigma) parameter of log of the distribution
   :param stdev: standard deviation of the distribution


