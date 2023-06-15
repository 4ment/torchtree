:py:mod:`torchtree.inference.mcmc.mcmc`
=======================================

.. py:module:: torchtree.inference.mcmc.mcmc


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.inference.mcmc.mcmc.MCMC




.. py:class:: MCMC(joint: torchtree.core.model.CallableModel, operators: List[torchtree.inference.mcmc.operator.MCMCOperator], iterations: int, **kwargs)


   Bases: :py:obj:`torchtree.core.serializable.JSONSerializable`, :py:obj:`torchtree.core.runnable.Runnable`

   Interface making an object JSON serializable.

   Serializable base class establishing
   :meth:`~torch.core.serializable.JSONSerializable.from_json` abstract method.

   .. py:method:: run() -> None


   .. py:method:: from_json(data: dict[str, any], dic: dict[str, any]) -> MCMC
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



