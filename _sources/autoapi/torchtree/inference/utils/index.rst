:py:mod:`torchtree.inference.utils`
===================================

.. py:module:: torchtree.inference.utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   torchtree.inference.utils.extract_tensors_and_parameters



.. py:function:: extract_tensors_and_parameters(params: list[Union[torchtree.core.parameter.Parameter, torchtree.core.parametric.Parametric]], dic: dict[str, any]) -> tuple[torchtree.typing.ListTensor, torchtree.typing.ListParameter]

   Parse a list containing parameters or objects inheriting from Parametric
   and return a tuple containing every tensor and their corresponding parameters.

   :param params: list of parameters or parametric objects
   :type params: list(Parameter or Parametric)
   :param dic: dictionary containing every instanciated objects
   :return: tensors and Parameters
   :rtype: list(list(Tensor), list(Parameter))


