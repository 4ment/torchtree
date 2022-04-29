from __future__ import annotations

from typing import Union

from torchtree.core.parameter import Parameter
from torchtree.core.parametric import Parametric
from torchtree.core.utils import JSONParseError, process_objects
from torchtree.typing import ListParameter, ListTensor


def extract_tensors_and_parameters(
    params: list[Union[Parameter, Parametric]], dic: dict[str, any]
) -> tuple[ListTensor, ListParameter]:
    r"""Parse a list containing parameters or objects inheriting from Parametric
    and return a tuple containing every tensor and their corresponding parameters.

    :param params: list of parameters or parametric objects
    :type params: list(Parameter or Parametric)
    :param dic: dictionary containing every instanciated objects
    :return: tensors and Parameters
    :rtype: list(list(Tensor), list(Parameter))
    """
    parameter_or_parametric_list = process_objects(params, dic)
    if not isinstance(parameter_or_parametric_list, list):
        parameter_or_parametric_list = [parameter_or_parametric_list]

    tensors = []
    parameters = []
    for poml in parameter_or_parametric_list:
        if isinstance(poml, Parameter):
            tensors.append(poml.tensor)
            parameters.append(poml)
        elif isinstance(poml, Parametric):
            for parameter in poml.parameters():
                tensors.append(parameter.tensor)
                parameters.append(parameter)
        else:
            raise JSONParseError(
                'Optimizable expects a list of Parameters or Parametric models\n{}'
                ' was provided'.format(type(poml))
            )
    return tensors, parameters
