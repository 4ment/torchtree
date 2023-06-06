:py:mod:`torchtree.evolution.bdsk`
==================================

.. py:module:: torchtree.evolution.bdsk


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.evolution.bdsk.BDSKModel
   torchtree.evolution.bdsk.PiecewiseConstantBirthDeath



Functions
~~~~~~~~~

.. autoapisummary::

   torchtree.evolution.bdsk.epidemiology_to_birth_death



.. py:function:: epidemiology_to_birth_death(R, delta, s, r=None)

   Convert epidemiology to birth death parameters.

   :param R: effective reproductive number
   :param delta: total rate of becoming non infectious
   :param s: probability of an individual being sampled
   :param r: removal probability
   :return: lambda, mu, psi


.. py:class:: BDSKModel(id_: torchtree.typing.ID, tree_model: torchtree.evolution.tree_model.TimeTreeModel, R: torchtree.core.abstractparameter.AbstractParameter, delta: torchtree.core.abstractparameter.AbstractParameter, s: torchtree.core.abstractparameter.AbstractParameter, rho: torchtree.core.abstractparameter.AbstractParameter = None, origin: torchtree.core.abstractparameter.AbstractParameter = None, origin_is_root_edge: bool = False, times: torchtree.core.abstractparameter.AbstractParameter = None, relative_times: bool = False, survival: bool = True, removal_probability: torchtree.core.abstractparameter.AbstractParameter = None)

   Bases: :py:obj:`torchtree.core.model.CallableModel`

   Birth–death skyline plot as a model for transmission.

   Effective population size :math:`R=\frac{\lambda}{\mu + \psi}`

   Total rate of becoming infectious :math:`\delta = \mu + \psi`

   Probability of being sampled :math:`s = \frac{\psi}{\mu + \psi}`

   :param R: effective reproductive number
   :param delta: total rate of becoming non infectious
   :param s: probability of an individual being sampled
   :param rho: probability of an individual being sampled at present
   :param origin: time at which the process starts (i.e. t_0)
   :param origin_is_root_edge: the origin is the branch above the root
   :param times: times of rate shift events
   :param relative_times: times are relative to origin
   :param survival: condition on observing at least one sample
   :param removal_probability: probability of an individual to become
     noninfectious immediately after sampling
   :param validate_args:

   .. py:method:: from_json(data, dic)
      :classmethod:



.. py:class:: PiecewiseConstantBirthDeath(lambda_: torch.Tensor, mu: torch.Tensor, psi: torch.Tensor, *, rho: torch.Tensor = torch.zeros(1), origin: torch.Tensor = None, origin_is_root_edge: bool = False, times: torch.Tensor = None, relative_times=False, survival: bool = True, removal_probability: torch.Tensor = None, validate_args=None)

   Bases: :py:obj:`torch.distributions.distribution.Distribution`

   Piecewise constant birth death model.

   :param lambda_: birth rates
   :param mu: death rates
   :param psi: sampling rates
   :param rho: sampling effort
   :param origin: time at which the process starts (i.e. t_0)
   :param origin_is_root_edge: the origin is the branch above the root
   :param times: times of rate shift events
   :param relative_times: times are relative to origin
   :param survival: condition on observing at least one sample
   :param removal_probability: probability of an individual to become
     noninfectious immediately after sampling
   :param validate_args:

   .. py:attribute:: arg_constraints

      

   .. py:attribute:: support

      

   .. py:method:: log_q(A, B, t, t_i)

      Probability density of lineage alive between time t and t_i gives
      rise to observed clade.


   .. py:method:: p0(A, B, t, t_i)


   .. py:method:: log_p(t, t_i, rho)

      Probability density of lineage alive between time t and t_i has no
      descendant at time t_m.


   .. py:method:: log_prob(node_heights: torch.Tensor)

      Returns the log of the probability density/mass function evaluated at
      `value`.

      Args:
          value (Tensor):



