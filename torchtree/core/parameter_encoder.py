import json

import torch

from torchtree.core.parameter import Parameter
from torchtree.core.utils import TensorEncoder


class ParameterEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return TensorEncoder.default(self, obj)
        elif isinstance(obj, Parameter):
            return {
                'id': obj.id,
                'type': 'torchtree.Parameter',
                'tensor': obj.tensor.tolist(),
                'dtype': str(obj.tensor.dtype),
                'nn': isinstance(obj.tensor, torch.nn.Parameter),
            }
        return json.JSONEncoder.default(self, obj)
