from torchtree.cli.utils import length_of_tensor_in_dict_parameter


def create_scaler_operator(id_, joint, parameters, arg):
    if isinstance(parameters, list) and len(parameters) > 1:
        parameter_ids = []
        length = 0
        for param in parameters:
            parameter_ids.append(param["id"])
            length += length_of_tensor_in_dict_parameter(param)
    else:
        if isinstance(parameters, list) and len(parameters) == 1:
            parameters = parameters[0]
        length = length_of_tensor_in_dict_parameter(parameters)
        parameter_ids = parameters["id"]

    operator = {
        "id": f"{id_}.operator",
        "type": "ScalerOperator",
        "joint": joint,
        "parameters": parameter_ids,
        "weight": float(length),
        "scaler": 0.001,
    }
    return operator


def create_sliding_window_operator(id_, joint, parameters, arg):
    if isinstance(parameters, list) and len(parameters) > 1:
        length = 0
        parameter_ids = []
        for param in parameters:
            parameter_ids.append(param["id"])
            length += length_of_tensor_in_dict_parameter(param)
    else:
        if isinstance(parameters, list) and len(parameters) == 1:
            parameters = parameters[0]
        length = length_of_tensor_in_dict_parameter(parameters)
        parameter_ids = parameters["id"]

    operator = {
        "id": f"{id_}.operator",
        "type": "SlidingWindowOperator",
        "joint": joint,
        "parameters": parameter_ids,
        "weight": float(length),
        "width": 0.5,
    }
    return operator
