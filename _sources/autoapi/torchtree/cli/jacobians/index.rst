:py:mod:`torchtree.cli.jacobians`
=================================

.. py:module:: torchtree.cli.jacobians


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   torchtree.cli.jacobians.create_jacobians



.. py:function:: create_jacobians(dict_def: dict[str, Any]) -> list[str]

   This function looks for parameters of type
   :class:`~torchtree.core.parameter.TransformedParameter` and returns their IDs.

   :param dict dict_def: dictionary containing model specification
   :return: IDs of the transformed parameters
   :rtype: list(str)


