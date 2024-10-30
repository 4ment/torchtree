torchtree.inference.mcmc.mcmc
=============================

.. py:module:: torchtree.inference.mcmc.mcmc


Classes
-------

.. autoapisummary::

   torchtree.inference.mcmc.mcmc.MCMC


Module Contents
---------------

.. py:class:: MCMC(id_: torchtree.typing.ID, joint: torchtree.core.model.CallableModel, operators: List[torchtree.inference.mcmc.operator.MCMCOperator], iterations: int, **kwargs)

   Bases: :py:obj:`torchtree.core.identifiable.Identifiable`, :py:obj:`torchtree.core.runnable.Runnable`


   Represents a Markov Chain Monte Carlo (MCMC) algorithm for inference.

   :param id_: The ID of the MCMC instance.
   :type id_: ID
   :param joint: The joint model used for inference.
   :type joint: CallableModel
   :param operators: The list of MCMC operators.
   :type operators: List[MCMCOperator]
   :param int iterations: The number of iterations for the MCMC algorithm.
   :param dict kwargs: Additional keyword arguments.


   .. py:attribute:: joint


   .. py:attribute:: iterations


   .. py:attribute:: loggers


   .. py:attribute:: checkpoint


   .. py:attribute:: checkpoint_frequency


   .. py:attribute:: every


   .. py:attribute:: parameters
      :value: []



   .. py:method:: run() -> None

      Run the MCMC algorithm.



   .. py:method:: state_dict() -> dict[str, Any]

      Returns the current state of the MCMC object as a dictionary.



   .. py:method:: load_state_dict(state_dict: dict[str, Any]) -> None

      Load the state dictionary into the MCMC algorithm.



   .. py:method:: save_full_state() -> None

      Save the full state of the MCMC algorithm.



   .. py:method:: from_json(data: dict[str, Any], dic: dict[str, Any]) -> MCMC
      :classmethod:


      Creates an MCMC instance from a dictionary.

      :param dict[str, Any] data: dictionary representation of a parameter object.
      :param dict[str, Identifiable] dic: dictionary containing torchtree objects
          keyed by their ID.

      **JSON attributes**:

       Mandatory:
        - id (str): identifier of object.
        - joint (str or dict): joint distribution of interest implementing CallableModel.
        - operators (list of dict): list of operators implementing MCMCOperator.
        - iterations (int): number of iterations.

       Optional:
        - loggers (list of dict): list of loggers implementing MCMCOperator.
        - checkpoint (bool or str): checkpoint file name (Default: checkpoint.json).
          No checkpointing if False is specified.
        - checkpoint_frequency (int): frequency of checkpointing (Default: 1000).
        - every (int): on-screen logging frequency (Default: 100).



