torchtree.distributions.one_on_x
================================

.. py:module:: torchtree.distributions.one_on_x

.. autoapi-nested-parse::

   One on X prior.



Classes
-------

.. autoapisummary::

   torchtree.distributions.one_on_x.OneOnX


Module Contents
---------------

.. py:class:: OneOnX(validate_args=None)

   Bases: :py:obj:`torch.distributions.Distribution`


   One on X prior.

   Calculates the (improper) prior proportional to
   :math:`\prod_i (1/x_i)` for the given statistic x.


   .. py:attribute:: arg_constraints

      Returns a dictionary from argument names to
      :class:`~torch.distributions.constraints.Constraint` objects that
      should be satisfied by each argument of this distribution. Args that
      are not tensors need not appear in this dict.


   .. py:attribute:: support

      Returns a :class:`~torch.distributions.constraints.Constraint` object
      representing this distribution's support.


   .. py:method:: log_prob(value: torch.Tensor) -> torch.Tensor

      Returns the log of the probability density/mass function evaluated at
      `value`.

      :param value:
      :type value: Tensor



