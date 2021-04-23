import numpy as np
import torch
import torch.distributions

from phylotorch.core.model import CallableModel
from phylotorch.core.utils import process_objects, process_object


class MultivariateNormal(CallableModel):

    def __init__(self, id_, x, loc, **kwargs):
        super(MultivariateNormal, self).__init__(id_)
        self.x = x
        self.loc = loc
        self.parameterization = list(kwargs.keys())[0]
        self.parameter = list(kwargs.values())[0]

        self.add_parameter(self.loc)
        self.add_parameter(self.parameter)

        if isinstance(self.x, (list, tuple)):
            for xx in self.x:
                self.add_parameter(xx)
        else:
            self.add_parameter(self.x)

    def rsample(self, sample_shape=torch.Size()):
        kwargs = {self.parameterization: self.parameter.tensor}
        x = torch.distributions.MultivariateNormal(self.loc.tensor, **kwargs).rsample(sample_shape)
        if isinstance(self.x, (list, tuple)):
            offset = 0
            for xx in self.x:
                xx.tensor = x[offset:(offset + xx.shape[0])]
                offset += xx.shape[0]
        else:
            self.x.tensor = x

    def sample(self, sample_shape=torch.Size()):
        kwargs = {self.parameterization: self.parameter.tensor}
        x = torch.distributions.MultivariateNormal(self.loc.tensor, **kwargs).sample(sample_shape)
        if isinstance(self.x, (list, tuple)):
            offset = 0
            for xx in self.x:
                xx.tensor = x[offset:(offset + xx.shape[0])]
                offset += xx.shape[0]
        else:
            self.x.tensor = x

    def log_prob(self):
        kwargs = {self.parameterization: self.parameter.tensor}
        if isinstance(self.x, (list, tuple)):
            return torch.distributions.MultivariateNormal(self.loc.tensor, **kwargs).log_prob(
                torch.cat([xx.tensor for xx in self.x]))
        else:
            return torch.distributions.MultivariateNormal(self.loc.tensor, **kwargs).log_prob(self.x.tensor)

    def update(self, value):
        pass

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        pass

    def _call(self, *args, **kwargs):
        return self.log_prob()

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        x = process_objects(data['x'], dic)
        loc = process_object(data['parameters']['loc'], dic)
        kwargs = {}

        for p in ('covariance_matrix', 'scale_tril', 'precision_matrix'):
            if p in data['parameters']:
                parameterization = p
                kwargs[parameterization] = process_object(data['parameters'][parameterization], dic)

        if len(kwargs) != 1:
            raise NotImplementedError(
                'MultivariateNormal is parameterized with either covariance_matrix, scale_tril or precision_matrix')

        if isinstance(x, list):
            x_count = np.sum([xx.shape[0] for xx in x])
            assert x_count == loc.shape[0]
            assert torch.Size((x_count, x_count)) == kwargs[parameterization].shape
        else:
            # event_shape must match
            assert x.shape[-1:] == loc.shape[-1:]
            assert x.shape[-1:] + x.shape[-1:] == kwargs[parameterization].shape[-2:]

        return cls(id_, x, loc, **kwargs)
