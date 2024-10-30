torchtree.evolution.birth_death
===============================

.. py:module:: torchtree.evolution.birth_death


Classes
-------

.. autoapisummary::

   torchtree.evolution.birth_death.BirthDeathModel
   torchtree.evolution.birth_death.BirthDeath


Module Contents
---------------

.. py:class:: BirthDeathModel(id_: torchtree.typing.ID, tree_model: torchtree.evolution.tree_model.TimeTreeModel, lambda_: torchtree.core.abstractparameter.AbstractParameter, mu: torchtree.core.abstractparameter.AbstractParameter, psi: torchtree.core.abstractparameter.AbstractParameter, rho: torchtree.core.abstractparameter.AbstractParameter, origin: torchtree.core.abstractparameter.AbstractParameter, survival: bool = True)

   Bases: :py:obj:`torchtree.core.model.CallableModel`


   Birth–death model

   :param lambda_: birth rate
   :param mu: death rate
   :param psi: sampling rate
   :param rho: sampling effort
   :param origin: time at which the process starts (i.e. t_0)
   :param survival: condition on observing at least one sample


   .. py:attribute:: tree_model


   .. py:attribute:: lambda_


   .. py:attribute:: mu


   .. py:attribute:: psi


   .. py:attribute:: rho


   .. py:attribute:: origin


   .. py:attribute:: survival


   .. py:method:: handle_model_changed(model, obj, index)


   .. py:method:: from_json(data, dic)
      :classmethod:


      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



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

      Returns a dictionary from argument names to
      :class:`~torch.distributions.constraints.Constraint` objects that
      should be satisfied by each argument of this distribution. Args that
      are not tensors need not appear in this dict.


   .. py:attribute:: lambda_


   .. py:attribute:: mu


   .. py:attribute:: psi


   .. py:attribute:: rho


   .. py:attribute:: origin


   .. py:attribute:: survival


   .. py:method:: log_q(A, B, t, t_i)

      Probability density of lineage alive between time t and t_i gives
      rise to observed clade.



   .. py:method:: log_p(t)

      Probability density of lineage alive between time t and t_i has no
      descendant at time t_m.



   .. py:method:: log_prob(node_heights: torch.Tensor)

      Returns the log of the probability density/mass function evaluated at
      `value`.

      :param value:
      :type value: Tensor



