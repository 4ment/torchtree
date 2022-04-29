from __future__ import annotations

from typing import Any


def create_jacobians(dict_def: dict[str, Any]) -> list[str]:
    """This function looks for parameters of type
    :class:`~torchtree.core.parameter.TransformedParameter` and returns their IDs.

    :param dict dict_def: dictionary containing model specification
    :return: IDs of the transformed parameters
    :rtype: list(str)
    """
    params = []
    if isinstance(dict_def, list):
        for element in dict_def:
            params.extend(create_jacobians(element))
    elif isinstance(dict_def, dict):
        if 'type' in dict_def and dict_def['type'] == 'TransformedParameter':
            if not (
                dict_def['transform'] == 'torch.distributions.AffineTransform'
                and dict_def['parameters']['scale'] == 1.0
            ):
                params.append(dict_def['id'])
        for value in dict_def.values():
            params.extend(create_jacobians(value))
    return params
