:py:mod:`torchtree.distributions.one_on_x`
==========================================

.. py:module:: torchtree.distributions.one_on_x


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.distributions.one_on_x.OneOnX




.. py:class:: OneOnX(validate_args=None)



   Distribution is the abstract base class for probability distributions.

   .. py:attribute:: arg_constraints

      

   .. py:attribute:: support

      

   .. py:method:: log_prob(value: torch.Tensor) -> torch.Tensor

      Returns the log of the probability density/mass function evaluated at
      `value`.

      Args:
          value (Tensor):



