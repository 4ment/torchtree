:py:mod:`torchtree.distributions.normal`
========================================

.. py:module:: torchtree.distributions.normal


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.distributions.normal.Normal




.. py:class:: Normal(loc: Union[float, torch.Tensor], scale: Union[float, torch.Tensor] = None, precision: Union[float, torch.Tensor] = None, validate_args=None)



   Creates a normal distribution parameterized by :attr:`loc` and
    :attr:`scale` or :attr:`precision`.

   :param loc: mean of the distribution (often referred to as mu)
   :param scale: standard deviation of the distribution (often referred to as sigma)
   :param precision: precision of the distribution (precision = 1/scale^2)


