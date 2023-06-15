:py:mod:`torchtree.variational.kl`
==================================

.. py:module:: torchtree.variational.kl


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.variational.kl.ELBO
   torchtree.variational.kl.KLpq
   torchtree.variational.kl.KLpqImportance
   torchtree.variational.kl.SELBO




.. py:class:: ELBO(id_: torchtree.typing.ID, q: torchtree.distributions.distributions.DistributionModel, p: torchtree.core.model.CallableModel, samples: torch.Size, entropy=False, score=False)


   Bases: :py:obj:`torchtree.core.model.CallableModel`

   Class representing the evidence lower bound (ELBO) objective.
   Maximizing the ELBO is equivalent to minimizing exclusive Kullback-Leibler
   divergence from p to q :math:`KL(q\|p)`.

   The shape of ``samples`` is at most 2 dimensional.

   - 0 or 1 dimension N or [N]: standard ELBO.
   - 2 dimensions [N,K]: multi sample ELBO.

   :param id_: ID of KLqp object.
   :type id_: str or None
   :param DistributionModel q: variational distribution.
   :param CallableModel p: joint distribution.
   :param torch.Size samples: number of samples.
   :param bool entropy: use entropy instead of Monte Carlo approximation
       for variational distribution
   :param bool score: use score function instead of pathwise gradient estimator

   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: from_json(data, dic) -> ELBO
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: KLpq(id_: torchtree.typing.ID, q: torchtree.distributions.distributions.DistributionModel, p: torchtree.core.model.CallableModel, samples: torch.Size)


   Bases: :py:obj:`torchtree.core.model.CallableModel`

   Calculate inclusive Kullback-Leibler divergence from q to p :math:`KL(p\|q)`
   using self-normalized importance sampling gradient estimator [#oh1992]_.

   :param id_: ID of KLpq object.
   :type id_: str or None
   :param DistributionModel q: variational distribution.
   :param CallableModel p: joint distribution.
   :param torch.Size samples: number of samples.

   .. [#oh1992] Oh, M.-S., & Berger, J. O. (1992). Adaptive importance sampling in
    Monte Carlo integration.
    Journal of Statistical Computation and Simulation, 41(3-4), 143–168.

   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: from_json(data, dic) -> KLpq
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: KLpqImportance(id_: torchtree.typing.ID, q: torchtree.distributions.distributions.DistributionModel, p: torchtree.core.model.CallableModel, samples: torch.Size)


   Bases: :py:obj:`torchtree.core.model.CallableModel`

   Class for minimizing inclusive Kullback-Leibler divergence
   from q to p :math:`KL(p\|q)`
   using self-normalized importance sampling gradient estimator [#oh1992]_.

   :param id_: ID of object.
   :type id_: str or None
   :param DistributionModel q: variational distribution.
   :param CallableModel p: joint distribution.
   :param torch.Size samples: number of samples.


   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: SELBO(id_: torchtree.typing.ID, components: list[torchtree.distributions.distributions.DistributionModel], weights: torchtree.core.abstractparameter.AbstractParameter, p: torchtree.core.model.CallableModel, samples: torch.Size, entropy=False)


   Bases: :py:obj:`torchtree.core.model.CallableModel`

   Class representing the stratified evidence lower bound (SELBO) objective.
   Maximizing the SELBO is equivalent to minimizing exclusive Kullback-Leibler
   divergence from p to q :math:`KL(q\|p)` where :math:`q=\sum_i \alpha_i q_i`.

   The shape of ``samples`` is at most 2 dimensional.

   - 0 or 1 dimension N or [N]: standard ELBO.
   - 2 dimensions [N,K]: multi sample ELBO.

   :param id_: ID of KLqp object.
   :type id_: str or None
   :param DistributionModel components: list of distribution.
   :param AbstractParameter weights:
   :param CallableModel p: joint distribution.
   :param torch.Size samples: number of samples.
   :param bool entropy: use entropy instead of Monte Carlo approximation
       for variational distribution

   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



