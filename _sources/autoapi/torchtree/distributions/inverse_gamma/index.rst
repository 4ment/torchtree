:py:mod:`torchtree.distributions.inverse_gamma`
===============================================

.. py:module:: torchtree.distributions.inverse_gamma


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.distributions.inverse_gamma.InverseGamma




.. py:class:: InverseGamma(concentration, rate, validate_args=None)



   Extension of the Distribution class, which applies a sequence of Transforms
   to a base distribution.  Let f be the composition of transforms applied::

       X ~ BaseDistribution
       Y = f(X) ~ TransformedDistribution(BaseDistribution, f)
       log p(Y) = log p(X) + log |det (dX/dY)|

   Note that the ``.event_shape`` of a :class:`TransformedDistribution` is the
   maximum shape of its base distribution and its transforms, since transforms
   can introduce correlations among events.

   An example for the usage of :class:`TransformedDistribution` would be::

       # Building a Logistic Distribution
       # X ~ Uniform(0, 1)
       # f = a + b * logit(X)
       # Y ~ f(X) ~ Logistic(a, b)
       base_distribution = Uniform(0, 1)
       transforms = [SigmoidTransform().inv, AffineTransform(loc=a, scale=b)]
       logistic = TransformedDistribution(base_distribution, transforms)

   For more examples, please look at the implementations of
   :class:`~torch.distributions.gumbel.Gumbel`,
   :class:`~torch.distributions.half_cauchy.HalfCauchy`,
   :class:`~torch.distributions.half_normal.HalfNormal`,
   :class:`~torch.distributions.log_normal.LogNormal`,
   :class:`~torch.distributions.pareto.Pareto`,
   :class:`~torch.distributions.weibull.Weibull`,
   :class:`~torch.distributions.relaxed_bernoulli.RelaxedBernoulli` and
   :class:`~torch.distributions.relaxed_categorical.RelaxedOneHotCategorical`

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



