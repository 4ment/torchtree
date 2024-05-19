:py:mod:`torchtree.evolution.coalescent`
========================================

.. py:module:: torchtree.evolution.coalescent


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.evolution.coalescent.AbstractCoalescentDistribution
   torchtree.evolution.coalescent.AbstractCoalescentModel
   torchtree.evolution.coalescent.ConstantCoalescentModel
   torchtree.evolution.coalescent.ConstantCoalescent
   torchtree.evolution.coalescent.ConstantCoalescentIntegratedModel
   torchtree.evolution.coalescent.ConstantCoalescentIntegrated
   torchtree.evolution.coalescent.ExponentialCoalescentModel
   torchtree.evolution.coalescent.ExponentialCoalescent
   torchtree.evolution.coalescent.PiecewiseConstantCoalescent
   torchtree.evolution.coalescent.PiecewiseConstantCoalescentModel
   torchtree.evolution.coalescent.PiecewiseConstantCoalescentGrid
   torchtree.evolution.coalescent.SoftPiecewiseConstantCoalescentGrid
   torchtree.evolution.coalescent.PiecewiseConstantCoalescentGridModel
   torchtree.evolution.coalescent.FakeTreeModel
   torchtree.evolution.coalescent.PiecewiseExponentialCoalescentGrid
   torchtree.evolution.coalescent.PiecewiseExponentialCoalescentGridModel
   torchtree.evolution.coalescent.PiecewiseLinearCoalescentGrid
   torchtree.evolution.coalescent.PiecewiseLinearCoalescentGridModel



Functions
~~~~~~~~~

.. autoapisummary::

   torchtree.evolution.coalescent.process_data_coalesent



.. py:class:: AbstractCoalescentDistribution(theta: torch.Tensor, validate_args=None)


   Bases: :py:obj:`torch.distributions.distribution.Distribution`

   Distribution is the abstract base class for probability distributions.

   .. py:attribute:: arg_constraints

      

   .. py:attribute:: support

      

   .. py:attribute:: has_rsample
      :value: False

      

   .. py:method:: maximum_likelihood(node_heights) -> torch.Tensor
      :classmethod:
      :abstractmethod:



.. py:class:: AbstractCoalescentModel(id_: torchtree.typing.ID, theta: torchtree.core.abstractparameter.AbstractParameter, tree_model: torchtree.evolution.tree_model.TimeTreeModel)


   Bases: :py:obj:`torchtree.core.model.CallableModel`

   Classes inheriting from :class:`Model` and
   :class:`collections.abc.Callable`.

   CallableModel are Callable and the returned value is cached in case
   we need to use this value multiple times without the need to
   recompute it.

   .. py:method:: distribution() -> AbstractCoalescentDistribution
      :abstractmethod:

      Returns underlying coalescent distribution.



.. py:class:: ConstantCoalescentModel(id_: torchtree.typing.ID, theta: torchtree.core.abstractparameter.AbstractParameter, tree_model: torchtree.evolution.tree_model.TimeTreeModel)


   Bases: :py:obj:`AbstractCoalescentModel`

   Classes inheriting from :class:`Model` and
   :class:`collections.abc.Callable`.

   CallableModel are Callable and the returned value is cached in case
   we need to use this value multiple times without the need to
   recompute it.

   .. py:method:: distribution() -> AbstractCoalescentDistribution

      Returns underlying coalescent distribution.


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: ConstantCoalescent(theta: torch.Tensor, validate_args=None)


   Bases: :py:obj:`AbstractCoalescentDistribution`

   Distribution is the abstract base class for probability distributions.

   .. py:attribute:: has_rsample
      :value: True

      

   .. py:method:: maximum_likelihood(node_heights) -> torch.Tensor
      :classmethod:


   .. py:method:: log_prob(node_heights: torch.Tensor) -> torch.Tensor

      Returns the log of the probability density/mass function evaluated at
      `value`.

      :param value:
      :type value: Tensor


   .. py:method:: rsample(sample_shape=torch.Size())

      Generates a sample_shape shaped reparameterized sample or sample_shape
      shaped batch of reparameterized samples if the distribution parameters
      are batched.



.. py:class:: ConstantCoalescentIntegratedModel(id_: torchtree.typing.ID, tree_model: torchtree.evolution.tree_model.TimeTreeModel, alpha: float, beta: float)


   Bases: :py:obj:`torchtree.core.model.CallableModel`

   Classes inheriting from :class:`Model` and
   :class:`collections.abc.Callable`.

   CallableModel are Callable and the returned value is cached in case
   we need to use this value multiple times without the need to
   recompute it.

   .. py:method:: distribution() -> AbstractCoalescentDistribution


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: ConstantCoalescentIntegrated(alpha: float, beta, validate_args=None)


   Bases: :py:obj:`torch.distributions.distribution.Distribution`

   Integrated Constant size coalescent/inverse gamma distribution.

   Integrate the product of constant population coalescent and inverse gamma distribtions
   with respect to population size.

   :param AbstractParameter theta: population size parameter
   :param float alpha: shape parameter of the gamma distribution.
   :param float beta: rate parameter of the gamma distribution.

   .. math::
      p(T; \alpha, \beta) &= \int_0^{\infty} p(\theta; \alpha, \beta) p(T \mid \theta) d\theta \\
                          &= \int_0^{\infty} \frac{\beta^\alpha}{\Gamma(\alpha)}\theta^{-\alpha-1} e^{-\beta/\theta} \theta^{-N} e^{-(\sum_{i=1}^N C_i t_i)/\theta} d\theta \\
                          &= \frac{\beta^\alpha}{\Gamma(\alpha)} \frac{\Gamma}{\left(\beta + \sum_{i=1}^N C_i t_i \right)^{\alpha + N}}

   The posterior distribution of the population size parameter is an inverse gamma with shape :math:`\alpha + N` and rate :math:`\beta + \sum_{i=1}^N C_i t_i`.

   .. py:attribute:: arg_constraints

      

   .. py:attribute:: support

      

   .. py:attribute:: has_rsample
      :value: False

      

   .. py:method:: log_prob(node_heights: torch.Tensor) -> torch.Tensor

      Returns the log of the probability density/mass function evaluated at
      `value`.

      :param value:
      :type value: Tensor



.. py:class:: ExponentialCoalescentModel(id_: torchtree.typing.ID, theta: torchtree.core.abstractparameter.AbstractParameter, growth: torchtree.core.abstractparameter.AbstractParameter, tree_model: torchtree.evolution.tree_model.TimeTreeModel)


   Bases: :py:obj:`AbstractCoalescentModel`

   Classes inheriting from :class:`Model` and
   :class:`collections.abc.Callable`.

   CallableModel are Callable and the returned value is cached in case
   we need to use this value multiple times without the need to
   recompute it.

   .. py:method:: distribution() -> AbstractCoalescentDistribution

      Returns underlying coalescent distribution.


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: ExponentialCoalescent(theta: torch.Tensor, growth: torch.Tensor, validate_args=None)


   Bases: :py:obj:`torch.distributions.distribution.Distribution`

   Distribution is the abstract base class for probability distributions.

   .. py:attribute:: arg_constraints

      

   .. py:attribute:: support

      

   .. py:attribute:: has_rsample
      :value: False

      

   .. py:method:: log_prob(node_heights: torch.Tensor) -> torch.Tensor

      Returns the log of the probability density/mass function evaluated at
      `value`.

      :param value:
      :type value: Tensor



.. py:class:: PiecewiseConstantCoalescent(theta: torch.Tensor, validate_args=None)


   Bases: :py:obj:`AbstractCoalescentDistribution`

   Distribution is the abstract base class for probability distributions.

   .. py:method:: sufficient_statistics(node_heights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]

      Returns sorted sufficient statistics and number of coalescent events
      per interval.

      This method is used by the block updating MCMC operator.

      :param torch.Tensor node_heights: node heights.
      :return: sufficient statistics and number of coalescent events
          per interval.
      :rtype tuple[torch.Tensor, torch.Tensor]


   .. py:method:: log_prob(node_heights: torch.Tensor) -> torch.Tensor

      Returns the log of the probability density/mass function evaluated at
      `value`.

      :param value:
      :type value: Tensor


   .. py:method:: maximum_likelihood(node_heights: torch.Tensor) -> torch.Tensor
      :classmethod:



.. py:class:: PiecewiseConstantCoalescentModel(id_: torchtree.typing.ID, theta: torchtree.core.abstractparameter.AbstractParameter, tree_model: torchtree.evolution.tree_model.TimeTreeModel)


   Bases: :py:obj:`AbstractCoalescentModel`

   Classes inheriting from :class:`Model` and
   :class:`collections.abc.Callable`.

   CallableModel are Callable and the returned value is cached in case
   we need to use this value multiple times without the need to
   recompute it.

   .. py:method:: distribution() -> AbstractCoalescentDistribution

      Returns underlying coalescent distribution.


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: PiecewiseConstantCoalescentGrid(thetas: torch.Tensor, grid: torch.Tensor, validate_args=None)


   Bases: :py:obj:`AbstractCoalescentDistribution`

   Distribution is the abstract base class for probability distributions.

   .. py:method:: sufficient_statistics(node_heights: torch.Tensor)


   .. py:method:: log_prob(node_heights: torch.Tensor) -> torch.Tensor

      Returns the log of the probability density/mass function evaluated at
      `value`.

      :param value:
      :type value: Tensor



.. py:class:: SoftPiecewiseConstantCoalescentGrid(thetas: torch.Tensor, grid: torch.Tensor, temperature: float = None, validate_args=None)


   Bases: :py:obj:`ConstantCoalescent`

   Distribution is the abstract base class for probability distributions.

   .. py:method:: log_prob(node_heights: torch.Tensor) -> torch.Tensor

      Returns the log of the probability density/mass function evaluated at
      `value`.

      :param value:
      :type value: Tensor



.. py:class:: PiecewiseConstantCoalescentGridModel(id_: torchtree.typing.ID, theta: torchtree.core.abstractparameter.AbstractParameter, grid: torchtree.core.abstractparameter.AbstractParameter, tree_model: torchtree.evolution.tree_model.TimeTreeModel, temperature: float = None)


   Bases: :py:obj:`AbstractCoalescentModel`

   Classes inheriting from :class:`Model` and
   :class:`collections.abc.Callable`.

   CallableModel are Callable and the returned value is cached in case
   we need to use this value multiple times without the need to
   recompute it.

   .. py:method:: distribution() -> AbstractCoalescentDistribution

      Returns underlying coalescent distribution.


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: FakeTreeModel(node_heights)


   .. py:property:: node_heights


   .. py:property:: sample_shape
      :type: torch.Size



.. py:function:: process_data_coalesent(data, dtype: torch.dtype) -> torchtree.core.abstractparameter.AbstractParameter


.. py:class:: PiecewiseExponentialCoalescentGrid(theta: torch.Tensor, growth: torch.Tensor, grid: torch.Tensor, validate_args=None)


   Bases: :py:obj:`torch.distributions.distribution.Distribution`

   Distribution is the abstract base class for probability distributions.

   .. py:attribute:: arg_constraints

      

   .. py:attribute:: support

      

   .. py:attribute:: has_rsample
      :value: False

      

   .. py:method:: log_prob(node_heights: torch.Tensor) -> torch.Tensor

      Returns the log of the probability density/mass function evaluated at
      `value`.

      :param value:
      :type value: Tensor



.. py:class:: PiecewiseExponentialCoalescentGridModel(id_: torchtree.typing.ID, theta: torchtree.core.abstractparameter.AbstractParameter, growth: torchtree.core.abstractparameter.AbstractParameter, grid: torchtree.core.abstractparameter.AbstractParameter, tree_model: torchtree.evolution.tree_model.TimeTreeModel)


   Bases: :py:obj:`AbstractCoalescentModel`

   Classes inheriting from :class:`Model` and
   :class:`collections.abc.Callable`.

   CallableModel are Callable and the returned value is cached in case
   we need to use this value multiple times without the need to
   recompute it.

   .. py:method:: distribution() -> AbstractCoalescentDistribution

      Returns underlying coalescent distribution.


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: PiecewiseLinearCoalescentGrid(theta: torch.Tensor, grid: torch.Tensor, validate_args=None)


   Bases: :py:obj:`torch.distributions.distribution.Distribution`

   Distribution is the abstract base class for probability distributions.

   .. py:attribute:: arg_constraints

      

   .. py:attribute:: support

      

   .. py:attribute:: has_rsample
      :value: False

      

   .. py:method:: log_prob(node_heights: torch.Tensor) -> torch.Tensor

      Returns the log of the probability density/mass function evaluated at
      `value`.

      :param value:
      :type value: Tensor



.. py:class:: PiecewiseLinearCoalescentGridModel(id_: torchtree.typing.ID, theta: torchtree.core.abstractparameter.AbstractParameter, grid: torchtree.core.abstractparameter.AbstractParameter, tree_model: torchtree.evolution.tree_model.TimeTreeModel)


   Bases: :py:obj:`AbstractCoalescentModel`

   Classes inheriting from :class:`Model` and
   :class:`collections.abc.Callable`.

   CallableModel are Callable and the returned value is cached in case
   we need to use this value multiple times without the need to
   recompute it.

   .. py:method:: distribution() -> AbstractCoalescentDistribution

      Returns underlying coalescent distribution.


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



