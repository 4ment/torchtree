:py:mod:`torchtree.core.parameter_utils`
========================================

.. py:module:: torchtree.core.parameter_utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   torchtree.core.parameter_utils.save_parameters
   torchtree.core.parameter_utils.pack_tensor



.. py:function:: save_parameters(file_name: str, parameters: list[torchtree.core.parameter.Parameter], safely=True)

   Save a list of parameters to a json file.

   :param str file_name: output file path
   :param parameters: list of parameters
   :type parameters: list(Parameter)
   :param bool safely: Create a temporary file if True


.. py:function:: pack_tensor(parameters: List[torchtree.core.parameter.Parameter], tensor: torch.Tensor) -> None

   Pack a tensor into a list of Parameter.


