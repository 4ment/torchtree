:py:mod:`torchtree.distributions.inverse_gamma`
===============================================

.. py:module:: torchtree.distributions.inverse_gamma

.. autoapi-nested-parse::

   Inverse gamma distribution parametrized by concentration and rate.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.distributions.inverse_gamma.InverseGamma




.. py:class:: InverseGamma(concentration, rate, validate_args=None)


   Bases: :py:obj:`torch.distributions.TransformedDistribution`

   Inverse gamma distribution parametrized by concentration and rate.

   Creates an inverse gamma distribution parameterized by :attr:`concentration`
   and :attr:`rate`.

   :param float or Tensor concentration: concentration parameter of the distribution
   :param float or Tensor rate: rate parameter of the distribution

   .. py:property:: concentration


   .. py:property:: rate


   .. py:attribute:: arg_constraints

      

   .. py:attribute:: support

      

   .. py:attribute:: has_rsample
      :value: True

      

   .. py:method:: expand(batch_shape, _instance=None)

      Returns a new distribution instance (or populates an existing instance
      provided by a derived class) with batch dimensions expanded to
      `batch_shape`. This method calls :class:`~torch.Tensor.expand` on
      the distribution's parameters. As such, this does not allocate new
      memory for the expanded distribution instance. Additionally,
      this does not repeat any args checking or parameter broadcasting in
      `__init__.py`, when an instance is first created.

      Args:
          batch_shape (torch.Size): the desired expanded size.
          _instance: new instance provided by subclasses that
              need to override `.expand`.

      Returns:
          New distribution instance with batch dimensions expanded to
          `batch_size`.



