:py:mod:`torchtree.evolution.coalescent`
========================================

.. py:module:: torchtree.evolution.coalescent


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.evolution.coalescent.AbstractCoalescentModel
   torchtree.evolution.coalescent.ConstantCoalescentModel
   torchtree.evolution.coalescent.AbstractCoalescentDistribution
   torchtree.evolution.coalescent.ConstantCoalescent
   torchtree.evolution.coalescent.ExponentialCoalescentModel
   torchtree.evolution.coalescent.ExponentialCoalescent
   torchtree.evolution.coalescent.PiecewiseConstantCoalescent
   torchtree.evolution.coalescent.PiecewiseConstantCoalescentModel
   torchtree.evolution.coalescent.PiecewiseConstantCoalescentGrid
   torchtree.evolution.coalescent.SoftPiecewiseConstantCoalescentGrid
   torchtree.evolution.coalescent.PiecewiseConstantCoalescentGridModel
   torchtree.evolution.coalescent.FakeTreeModel
   torchtree.evolution.coalescent.PiecewiseExponentialCoalescentGrid



Functions
~~~~~~~~~

.. autoapisummary::

   torchtree.evolution.coalescent.process_data_coalesent



.. py:class:: AbstractCoalescentModel(id_: torchtree.typing.ID, theta: torchtree.core.abstractparameter.AbstractParameter, tree_model: torchtree.evolution.tree_model.TimeTreeModel = None)



   Classes inheriting from :class:`Model` and
   :class:`collections.abc.Callable`.

   CallableModel are Callable and the returned value is cached in case
   we need to use this value multiple times without the need to
   recompute it.


.. py:class:: ConstantCoalescentModel(id_: torchtree.typing.ID, theta: torchtree.core.abstractparameter.AbstractParameter, tree_model: torchtree.evolution.tree_model.TimeTreeModel = None)



   Classes inheriting from :class:`Model` and
   :class:`collections.abc.Callable`.

   CallableModel are Callable and the returned value is cached in case
   we need to use this value multiple times without the need to
   recompute it.

   .. py:method:: from_json(data, dic)
      :classmethod:



.. py:class:: AbstractCoalescentDistribution(theta: torch.Tensor, validate_args=None)



   Distribution is the abstract base class for probability distributions.

   .. py:attribute:: arg_constraints

      

   .. py:attribute:: support

      

   .. py:attribute:: has_rsample
      :value: False

      

   .. py:method:: maximum_likelihood(node_heights) -> torch.Tensor
      :classmethod:
      :abstractmethod:



.. py:class:: ConstantCoalescent(theta: torch.Tensor, validate_args=None)



   Distribution is the abstract base class for probability distributions.

   .. py:attribute:: has_rsample
      :value: True

      

   .. py:method:: maximum_likelihood(node_heights) -> torch.Tensor
      :classmethod:


   .. py:method:: log_prob(node_heights: torch.Tensor) -> torch.Tensor

      Returns the log of the probability density/mass function evaluated at
      `value`.

      Args:
          value (Tensor):


   .. py:method:: rsample(sample_shape=torch.Size())

      Generates a sample_shape shaped reparameterized sample or sample_shape
      shaped batch of reparameterized samples if the distribution parameters
      are batched.



.. py:class:: ExponentialCoalescentModel(id_: torchtree.typing.ID, theta: torchtree.core.abstractparameter.AbstractParameter, growth: torchtree.core.abstractparameter.AbstractParameter, tree_model: torchtree.evolution.tree_model.TimeTreeModel = None)



   Classes inheriting from :class:`Model` and
   :class:`collections.abc.Callable`.

   CallableModel are Callable and the returned value is cached in case
   we need to use this value multiple times without the need to
   recompute it.

   .. py:method:: from_json(data, dic)
      :classmethod:



.. py:class:: ExponentialCoalescent(theta: torch.Tensor, growth: torch.Tensor, validate_args=None)



   Distribution is the abstract base class for probability distributions.

   .. py:attribute:: arg_constraints

      

   .. py:attribute:: support

      

   .. py:attribute:: has_rsample
      :value: False

      

   .. py:method:: log_prob(node_heights: torch.Tensor) -> torch.Tensor

      Returns the log of the probability density/mass function evaluated at
      `value`.

      Args:
          value (Tensor):



.. py:class:: PiecewiseConstantCoalescent(theta: torch.Tensor, validate_args=None)



   Distribution is the abstract base class for probability distributions.

   .. py:method:: log_prob(node_heights: torch.Tensor) -> torch.Tensor

      Returns the log of the probability density/mass function evaluated at
      `value`.

      Args:
          value (Tensor):


   .. py:method:: maximum_likelihood(node_heights: torch.Tensor) -> torch.Tensor
      :classmethod:



.. py:class:: PiecewiseConstantCoalescentModel(id_: torchtree.typing.ID, theta: torchtree.core.abstractparameter.AbstractParameter, tree_model: torchtree.evolution.tree_model.TimeTreeModel = None)



   Classes inheriting from :class:`Model` and
   :class:`collections.abc.Callable`.

   CallableModel are Callable and the returned value is cached in case
   we need to use this value multiple times without the need to
   recompute it.


.. py:class:: PiecewiseConstantCoalescentGrid(thetas: torch.Tensor, grid: torch.Tensor, validate_args=None)



   Distribution is the abstract base class for probability distributions.

   .. py:method:: log_prob(node_heights: torch.Tensor) -> torch.Tensor

      Returns the log of the probability density/mass function evaluated at
      `value`.

      Args:
          value (Tensor):



.. py:class:: SoftPiecewiseConstantCoalescentGrid(thetas: torch.Tensor, grid: torch.Tensor, temperature: float = None, validate_args=None)



   Distribution is the abstract base class for probability distributions.

   .. py:method:: log_prob(node_heights: torch.Tensor) -> torch.Tensor

      Returns the log of the probability density/mass function evaluated at
      `value`.

      Args:
          value (Tensor):



.. py:class:: PiecewiseConstantCoalescentGridModel(id_: torchtree.typing.ID, theta: torchtree.core.abstractparameter.AbstractParameter, grid: torchtree.core.abstractparameter.AbstractParameter, tree_model: torchtree.evolution.tree_model.TimeTreeModel = None, temperature: float = None)



   Classes inheriting from :class:`Model` and
   :class:`collections.abc.Callable`.

   CallableModel are Callable and the returned value is cached in case
   we need to use this value multiple times without the need to
   recompute it.

   .. py:method:: from_json(data, dic)
      :classmethod:



.. py:class:: FakeTreeModel(node_heights)

   .. py:property:: node_heights



.. py:function:: process_data_coalesent(data, dtype: torch.dtype) -> torchtree.core.abstractparameter.AbstractParameter


.. py:class:: PiecewiseExponentialCoalescentGrid(theta: torch.Tensor, growth: torch.Tensor, grid: torch.Tensor, validate_args=None)



   Distribution is the abstract base class for probability distributions.

   .. py:attribute:: arg_constraints

      

   .. py:attribute:: support

      

   .. py:attribute:: has_rsample
      :value: False

      

   .. py:method:: log_prob(node_heights: torch.Tensor) -> torch.Tensor

      Returns the log of the probability density/mass function evaluated at
      `value`.

      Args:
          value (Tensor):



