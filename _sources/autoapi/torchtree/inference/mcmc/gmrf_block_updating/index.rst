:py:mod:`torchtree.inference.mcmc.gmrf_block_updating`
======================================================

.. py:module:: torchtree.inference.mcmc.gmrf_block_updating


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.inference.mcmc.gmrf_block_updating.GMRFPiecewiseCoalescentBlockUpdatingOperator




.. py:class:: GMRFPiecewiseCoalescentBlockUpdatingOperator(id_: torchtree.typing.ID, coalescent: torchtree.evolution.coalescent.AbstractCoalescentModel, gmrf: torchtree.distributions.gmrf.GMRF, weight: float, target_acceptance_probability: float, scaler: float, **kwargs)


   Bases: :py:obj:`torchtree.inference.mcmc.operator.MCMCOperator`

   Class implementing the block-updating Markov chain Monte Carlo sampling
   for Gaussian Markov random fields (GMRF).

   :param ID id_: identifier of object.
   :param coalescent: coalescent object.
   :param GMRF gmrf: GMRF object.
   :param float weight: operator weight.
   :param float target_acceptance_probability: target acceptance probability.
   :param float scaler: scaler for tuning the precision parameter proposal.

   .. py:property:: tuning_parameter
      :type: float


   .. py:method:: adaptable_parameter() -> float


   .. py:method:: set_adaptable_parameter(value: float) -> None


   .. py:method:: propose_precision()


   .. py:method:: jacobian(wNative, gamma, precision_matrix)


   .. py:method:: gradient(numCoalEv, wNative, gamma, precision_matrix)


   .. py:method:: newton_raphson(numCoalEv, wNative, gamma, precision_matrix)


   .. py:method:: from_json(data: dict[str, Any], dic: dict[str, torchtree.core.identifiable.Identifiable]) -> GMRFPiecewiseCoalescentBlockUpdatingOperator
      :classmethod:

      Creates a GMRFPiecewiseCoalescentBlockUpdatingOperator object from a
      dictionary.

      :param dict[str, Any] data: dictionary representation of a GMRFCovariate
          object.
      :param dict[str, Identifiable] dic: dictionary containing torchtree objects
          keyed by their ID.

      **JSON attributes**:

       Mandatory:
        - id (str): unique string identifier.
        - coalescent (dict or str): coalescent model.
        - gmrf (dict or str): GMRF model.

       Optional:
        - weight (float): weight of operator (Default: 1)
        - scaler (float): rescale by root height (Default: 2.0).
        - target_acceptance_probability (float): target acceptance
          probability (Default: 0.24).
        - disable_adaptation (bool): disable adaptation (Default: false).

      .. note::
          The precision proposal is not tuned if the scaler is equal to 1.



