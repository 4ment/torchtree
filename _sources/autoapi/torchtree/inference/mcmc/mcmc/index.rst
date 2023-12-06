:py:mod:`torchtree.inference.mcmc.mcmc`
=======================================

.. py:module:: torchtree.inference.mcmc.mcmc


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.inference.mcmc.mcmc.MCMC




.. py:class:: MCMC(id_: torchtree.typing.ID, joint: torchtree.core.model.CallableModel, operators: List[torchtree.inference.mcmc.operator.MCMCOperator], iterations: int, **kwargs)


   Bases: :py:obj:`torchtree.core.identifiable.Identifiable`, :py:obj:`torchtree.core.runnable.Runnable`

   Abstract class making an object identifiable.

   :param str or None id_: identifier of object

   .. py:method:: run() -> None


   .. py:method:: state_dict() -> dict[str, Any]


   .. py:method:: load_state_dict(state_dict: dict[str, Any]) -> None


   .. py:method:: save_full_state() -> None


   .. py:method:: from_json(data: dict[str, Any], dic: dict[str, Any]) -> MCMC
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



