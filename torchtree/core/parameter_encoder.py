import json

import torch

from .parameter import Parameter


class ParameterEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            pass
        elif isinstance(obj, Parameter):
            return {
                'id': obj.id,
                'type': 'torchtree.Parameter',
                'tensor': obj.tensor.tolist(),
                'dtype': str(obj.tensor.dtype),
                'nn': isinstance(obj.tensor, torch.nn.Parameter),
            }
        return json.JSONEncoder.default(self, obj)
