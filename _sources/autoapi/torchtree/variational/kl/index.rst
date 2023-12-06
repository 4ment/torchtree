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

   The ELBO is defined as

   .. math::
       \mathcal{L}(q) = \mathbb{E}_q[\log(p(z, x)] - \mathbb{E}_q[\log(q(z; \phi))]

   Maximizing the ELBO wrt variational parameters :math:`\phi` is equivalent
   to minimizing the exclusive Kullback-Leibler divergence from the
   posterior distribution :math:`p` to the variational distribution :math:`q`
   :math:`\text{KL}(q\|p) = \mathbb{E}_q[\log q(z; \phi)]-\mathbb{E}_q[\log p(z| x)]`.

   The shape of ``samples`` is at most 2 dimensional.

   - 0 or 1 dimension N or [N]: standard ELBO.
   - 2 dimensions [N,K]: multi sample ELBO.

   :param id_: ID of ELBO object.
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

   Calculate inclusive Kullback-Leibler divergence from q to p
   :math:`\text{KL}(p\|q)` using self-normalized importance sampling
   gradient estimator.

   The self-normalized importance sampling :footcite:p:`murphy2012machine` estimate
   of :math:`\text{KL}(p \|q)` using the instrument distribution :math:`q` is

   .. math::
       \widehat{KL}(p||q) & = \sum_{s=1}^S \log\left(\frac{p(\tilde{z}_s | x)}
         {q(\tilde{z}_s ; \phi)}\right) w_s , \quad \tilde{z}_s \sim q(z; \phi) \\
       & \propto \sum_{s=1}^S \log\left(\frac{p(\tilde{z}_s)}
         {q(\tilde{z}_s;\phi)}\right) w_s

   where

   .. math::
       w_s = \frac{p(\tilde{z}_s, D, \tau)}{ q(\tilde{z}_s; \phi)} /
         \sum_{i=1}^N \frac{p(\tilde{z}_i, D, \tau)}{q(\tilde{z}_i; \phi)}.


   :param id_: ID of KLpq object.
   :type id_: str or None
   :param DistributionModel q: variational distribution.
   :param CallableModel p: joint distribution.
   :param torch.Size samples: number of samples.

   .. footbibliography::

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
   from q to p :math:`\text{KL}(p\|q)` using self-normalized importance
   sampling gradient estimator.

   .. math::
       \nabla \widehat{\text{KL}}(p\|q) = -\sum_{s=1}^S w_s
         \nabla\log q(\tilde{z}_s ; \phi) , \quad \tilde{z}_s \sim q(z; \phi)

   where

   .. math::
       w_s = \frac{p(\tilde{z}_s, D, \tau)}{ q(\tilde{z}_s; \phi)} /
         \sum_{i=1}^N \frac{p(\tilde{z}_i, D, \tau)}{q(\tilde{z}_i; \phi)}.

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
   divergence from p to q :math:`\text{KL}(q\|p)` where :math:`q=\sum_i \alpha_i q_i`
   :footcite:p:`morningstar2021automatic`.

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

   .. footbibliography::

   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



