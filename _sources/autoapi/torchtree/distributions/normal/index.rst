torchtree.distributions.normal
==============================

.. py:module:: torchtree.distributions.normal

.. autoapi-nested-parse::

   Normal distribution parametrized by location and precision.



Classes
-------

.. autoapisummary::

   torchtree.distributions.normal.Normal


Module Contents
---------------

.. py:class:: Normal(loc: Union[float, torch.Tensor], scale: Union[float, torch.Tensor, None] = None, precision: Union[float, torch.Tensor, None] = None, validate_args=None)

   Bases: :py:obj:`torch.distributions.Normal`


   Normal distribution parametrized by location and precision.

   Creates a normal distribution parameterized by :attr:`loc` and
   :attr:`scale` or :attr:`precision`.

   .. math:: X \sim \mathcal{N}(\mu, 1/\tau)

   where :math:`\tau = 1/ \sigma^2`

   :example:
   >>> x = torch.tensor(0.1)
   >>> scale = torch.tensor(0.1)
   >>> norm1 = Normal(torch.tensor(0.5), precision=1.0/scale**2)
   >>> norm2 = torch.distributions.Normal(torch.tensor(0.5), scale=scale)
   >>> norm1.log_prob(x) == norm2.log_prob(x)
   tensor(True)

   :param float or Tensor loc: mean of the distribution.
   :param float or Tensor scale: standard deviation.
   :param float or Tensor precision: precision.


