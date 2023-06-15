:py:mod:`torchtree.evolution.tree_likelihood`
=============================================

.. py:module:: torchtree.evolution.tree_likelihood


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.evolution.tree_likelihood.TreeLikelihoodModel



Functions
~~~~~~~~~

.. autoapisummary::

   torchtree.evolution.tree_likelihood.calculate_treelikelihood
   torchtree.evolution.tree_likelihood.calculate_treelikelihood_discrete
   torchtree.evolution.tree_likelihood.calculate_treelikelihood_tip_states_discrete
   torchtree.evolution.tree_likelihood.calculate_treelikelihood_discrete_safe
   torchtree.evolution.tree_likelihood.calculate_treelikelihood_discrete_rescaled
   torchtree.evolution.tree_likelihood.calculate_treelikelihood_tip_states_discrete_rescaled



.. py:function:: calculate_treelikelihood(partials: list, weights: torch.Tensor, post_indexing: list, mats: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor

   Simple function for calculating the log tree likelihood.

   :param partials: list of tensors of partials [S,N] leaves and [...,S,N] internals
   :param weights: [N]
   :param post_indexing: list of indexes in postorder
   :param mats: tensor of probability matrices [...,B,S,S]
   :param freqs: tensor of frequencies [...,S]
   :return: tree log likelihood [batch]


.. py:function:: calculate_treelikelihood_discrete(partials: list, weights: torch.Tensor, post_indexing: list, mats: torch.Tensor, freqs: torch.Tensor, props: torch.Tensor) -> torch.Tensor

   Calculate log tree likelihood with rate categories

   number of tips: T,
   number of internal nodes: I=T-1,
   number of branches: B=2T-2,
   number of states: S,
   number of sites: N,
   number of rate categories: K.

   The shape of internal partials after peeling is [...,K,S,N].

   :param partials: list of T tip partial tensors [S,N] and I internals [None]
   :param weights: [N]
   :param post_indexing: list of indexes in postorder
   :param mats: tensor of probability matrices [...,B,K,S,S]
   :param freqs: tensor of frequencies [...,1,S]
   :param props: tensor of proportions [...,K,1,1]
   :return: tree log likelihood [batch]


.. py:function:: calculate_treelikelihood_tip_states_discrete(partials: list, weights: torch.Tensor, post_indexing: list, mats: torch.Tensor, freqs: torch.Tensor, props: torch.Tensor) -> torch.Tensor

   Calculate log tree likelihood with rate categories using tip states

   number of tips: T,
   number of internal nodes: I=T-1,
   number of branches: B=2T-2,
   number of states: S,
   number of sites: N,
   number of rate categories: K.

   The shape of internal partials after peeling is [...,K,S,N].

   :param partials: list of T tip state tensors [N] and I internals [None]
   :param weights: [N]
   :param post_indexing: list of indexes in postorder
   :param mats: tensor of probability matrices [...,B,K,S,S]
   :param freqs: tensor of frequencies [...,1,S]
   :param props: tensor of proportions [...,K,1,1]
   :return: tree log likelihood [batch]


.. py:function:: calculate_treelikelihood_discrete_safe(partials: list, weights: torch.Tensor, post_indexing: list, mats: torch.Tensor, freqs: torch.Tensor, props: torch.Tensor, threshold: float) -> torch.Tensor

   Calculate log tree likelihood with rate categories using rescaling.

   This function is used when an underflow is detected for the first time (i.e. inf)
   since it is not recalculating partials that are above the threshold.

   :param partials: list of tensors of partials [S,N] leaves and [...,K,S,N] internals
   :param weights: [N]
   :param post_indexing:
   :param mats: tensor of matrices [...,B,K,S,S]
   :param freqs: tensor of frequencies [...,1,S]
   :param props: tensor of proportions [...,K,1,1]
   :param threshold: threshold for rescaling
   :return: tree log likelihood [batch]


.. py:function:: calculate_treelikelihood_discrete_rescaled(partials: list, weights: torch.Tensor, post_indexing: list, mats: torch.Tensor, freqs: torch.Tensor, props: torch.Tensor) -> torch.Tensor

   Calculate log tree likelihood with rate categories using rescaling

   :param partials: list of tensors of partials [S,N] leaves and [...,K,S,N] internals
   :param weights: [N]
   :param post_indexing:
   :param mats: tensor of matrices [...,B,K,S,S]
   :param freqs: tensor of frequencies [...,1,S]
   :param props: tensor of proportions [...,K,1,1]
   :return: tree log likelihood [batch]


.. py:function:: calculate_treelikelihood_tip_states_discrete_rescaled(partials: list, weights: torch.Tensor, post_indexing: list, mats: torch.Tensor, freqs: torch.Tensor, props: torch.Tensor) -> torch.Tensor

   Calculate rescaled log tree likelihood with rate categories using tip states
   and rescaling.

   :param partials: list of tensors of tip states [N] leaves and [...,K,S,N] internals
   :param weights: [N]
   :param post_indexing:
   :param mats: tensor of matrices [...,B,K,S,S]
   :param freqs: tensor of frequencies [...,1,S]
   :param props: tensor of proportions [...,K,1,1]
   :return: tree log likelihood [batch]


.. py:class:: TreeLikelihoodModel(id_: torchtree.typing.ID, site_pattern: torchtree.evolution.site_pattern.SitePattern, tree_model: torchtree.evolution.tree_model.TreeModel, subst_model: torchtree.evolution.substitution_model.abstract.SubstitutionModel, site_model: torchtree.evolution.site_model.SiteModel, clock_model: torchtree.evolution.branch_model.BranchModel = None, use_ambiguities=False, use_tip_states=False)


   Bases: :py:obj:`torchtree.core.model.CallableModel`

   Classes inheriting from :class:`Model` and
   :class:`collections.abc.Callable`.

   CallableModel are Callable and the returned value is cached in case
   we need to use this value multiple times without the need to
   recompute it.

   .. py:method:: calculate_with_tip_partials(mats, frequencies, probs)


   .. py:method:: calculate_with_tip_states(mats, frequencies, probs)


   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



