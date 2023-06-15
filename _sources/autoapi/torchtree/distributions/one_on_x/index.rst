:py:mod:`torchtree.distributions.one_on_x`
==========================================

.. py:module:: torchtree.distributions.one_on_x

.. autoapi-nested-parse::

   One on X prior.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.distributions.one_on_x.OneOnX




.. py:class:: OneOnX(validate_args=None)


   Bases: :py:obj:`torch.distributions.Distribution`

   One on X prior.

   Calculates the (improper) prior proportional to
   :math:`\prod_i (1/x_i)` for the given statistic x.

   .. py:attribute:: arg_constraints

      

   .. py:attribute:: support

      

   .. py:method:: log_prob(value: torch.Tensor) -> torch.Tensor

      Returns the log of the probability density/mass function evaluated at
      `value`.

      Args:
          value (Tensor):



