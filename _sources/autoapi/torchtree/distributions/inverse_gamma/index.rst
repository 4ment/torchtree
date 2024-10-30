torchtree.distributions.inverse_gamma
=====================================

.. py:module:: torchtree.distributions.inverse_gamma

.. autoapi-nested-parse::

   Inverse gamma distribution parametrized by concentration and rate.



Classes
-------

.. autoapisummary::

   torchtree.distributions.inverse_gamma.InverseGamma


Module Contents
---------------

.. py:class:: InverseGamma(concentration, rate, validate_args=None)

   Bases: :py:obj:`torch.distributions.TransformedDistribution`


   Inverse gamma distribution parametrized by concentration and rate.

   Creates an inverse gamma distribution parameterized by :attr:`concentration`
   and :attr:`rate`.

   :param float or Tensor concentration: concentration parameter of the distribution
   :param float or Tensor rate: rate parameter of the distribution


   .. py:attribute:: arg_constraints

      Returns a dictionary from argument names to
      :class:`~torch.distributions.constraints.Constraint` objects that
      should be satisfied by each argument of this distribution. Args that
      are not tensors need not appear in this dict.


   .. py:attribute:: support

      Returns a :class:`~torch.distributions.constraints.Constraint` object
      representing this distribution's support.


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

      :param batch_shape: the desired expanded size.
      :type batch_shape: torch.Size
      :param _instance: new instance provided by subclasses that
                        need to override `.expand`.

      :returns: New distribution instance with batch dimensions expanded to
                `batch_size`.



   .. py:property:: concentration


   .. py:property:: rate


