:py:mod:`torchtree.distributions.log_normal`
============================================

.. py:module:: torchtree.distributions.log_normal

.. autoapi-nested-parse::

   Lognormal distribution parametrized by mean and scale.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.distributions.log_normal.LogNormal




.. py:class:: LogNormal(mean: Union[torch.Tensor, float], scale: Union[torch.Tensor, float] = None, scale_real: Union[torch.Tensor, float] = None, validate_args=None)

   Bases: :py:obj:`torch.distributions.LogNormal`

   Lognormal distribution parametrized by mean and scale.

   Creates a lognormal distribution parameterized by :attr:`mean` and
   :attr:`scale`.

   :param mean: mean of the distribution
   :param scale: standard deviation of log of the distribution
   :param scale_real: standard deviation of the distribution


