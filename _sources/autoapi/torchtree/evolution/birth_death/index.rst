:py:mod:`torchtree.evolution.birth_death`
=========================================

.. py:module:: torchtree.evolution.birth_death


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.evolution.birth_death.BirthDeathModel
   torchtree.evolution.birth_death.BirthDeath




.. py:class:: BirthDeathModel(id_: torchtree.typing.ID, tree_model: torchtree.evolution.tree_model.TimeTreeModel, lambda_: torchtree.core.abstractparameter.AbstractParameter, mu: torchtree.core.abstractparameter.AbstractParameter, psi: torchtree.core.abstractparameter.AbstractParameter, rho: torchtree.core.abstractparameter.AbstractParameter, origin: torchtree.core.abstractparameter.AbstractParameter, survival: bool = True)

   Bases: :py:obj:`torchtree.core.model.CallableModel`

   Birth–death model

   :param lambda_: birth rate
   :param mu: death rate
   :param psi: sampling rate
   :param rho: sampling effort
   :param origin: time at which the process starts (i.e. t_0)
   :param survival: condition on observing at least one sample

   .. py:method:: handle_model_changed(model, obj, index)


   .. py:method:: from_json(data, dic)
      :classmethod:



.. py:class:: BirthDeath(lambda_: torch.Tensor, mu: torch.Tensor, psi: torch.Tensor, rho: torch.Tensor, origin: torch.Tensor, survival: bool = True, validate_args=None)

   Bases: :py:obj:`torch.distributions.distribution.Distribution`

   Constant birth death model

   :param lambda_: birth rate
   :param mu: death rate
   :param psi: sampling rate
   :param rho: sampling effort
   :param origin: time at which the process starts (i.e. t_0)
   :param survival: condition on observing at least one sample
   :param validate_args:

   .. py:attribute:: arg_constraints

      

   .. py:method:: log_q(A, B, t, t_i)

      Probability density of lineage alive between time t and t_i gives
      rise to observed clade.


   .. py:method:: log_p(t)

      Probability density of lineage alive between time t and t_i has no
      descendant at time t_m.


   .. py:method:: log_prob(node_heights: torch.Tensor)

      Returns the log of the probability density/mass function evaluated at
      `value`.

      Args:
          value (Tensor):



