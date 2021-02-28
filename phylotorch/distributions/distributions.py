import inspect
from collections import OrderedDict

import torch

from ..core.model import CallableModel, Parameter
from ..core.utils import get_class, process_object


class Distribution(CallableModel):

    def __init__(self, id_, dist, x, args, **kwargs):
        super(Distribution, self).__init__(id_)
        self.dist = dist
        self.x = x
        self.args = args
        self.kwargs = kwargs

        for p in self.args.values():
            if p.id is not None:
                self.add_parameter(p)
        self.add_parameter(self.x)

    # def rsample(self, sample_shape=torch.Size()):
    #     return self.dist(self.args.keys(), **self.kwargs).rsample(sample_shape)

    def rsample(self):
        x = self.dist(*[arg.tensor for arg in self.args.values()],
                      **self.kwargs).rsample()
        self.x.tensor = x

    def sample(self):
        x = self.dist(*[arg.tensor for arg in self.args.values()],
                      **self.kwargs).sample()
        self.x.tensor = x

    def log_prob(self):
        return self.dist(*[arg.tensor for arg in self.args.values()],
                         **self.kwargs).log_prob(self.x.tensor)

    def update(self, value):
        for name in self.args.keys():
            if name in value:
                self.args[name] = value[name]

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        pass

    def __call__(self, *args, **kwargs):
        return self.log_prob()

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        klass = get_class(data['distribution'])
        x = process_object(data['x'], dic)

        signature_params = list(inspect.signature(klass.__init__).parameters)
        params = OrderedDict()
        if 'parameters' not in data:
            return cls(id_, klass, x, {})

        data_dist = data['parameters']
        for arg in signature_params[1:]:
            if arg in data_dist:
                if isinstance(data_dist[arg], str):
                    params[arg] = dic[data_dist[arg]]
                # elif isinstance(data_dist[arg], float):
                #     params[arg] = Parameter(None, torch.tensor([data_dist[arg]], dtype=x.dtype))
                elif isinstance(data_dist[arg], list):
                    params[arg] = Parameter(None, torch.tensor(data_dist[arg], dtype=x.dtype))
                else:
                    params[arg] = process_object(data_dist[arg], dic)
                # else:
                #     id = data[arg]['id']
                #     params[id] = as_tensor(data[arg])

        return cls(id_, klass, x, params)
