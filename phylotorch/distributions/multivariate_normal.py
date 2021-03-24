import numpy as np
import torch
import torch.distributions

from ..core.model import CallableModel
from ..core.utils import process_objects, process_object


class MultivariateNormal(CallableModel):

    def __init__(self, id_, x, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None):
        super(MultivariateNormal, self).__init__(id_)
        self.x = x
        self.loc = loc
        self.covariance_matrix = covariance_matrix
        self.precision_matrix = precision_matrix
        self.scale_tril = scale_tril

        self.add_parameter(self.loc)
        if covariance_matrix is not None:
            self.add_parameter(covariance_matrix)
        if precision_matrix is not None:
            self.add_parameter(precision_matrix)

        if isinstance(self.x, list):
            for xx in self.x:
                self.add_parameter(xx)
        else:
            self.add_parameter(self.x)

    def create_tril(self):
        tril = torch.zeros((self.loc.shape[0], self.loc.shape[0]), dtype=self.loc.dtype)
        tril_indices = torch.tril_indices(row=self.loc.shape[0], col=self.loc.shape[0], offset=0)
        tril[tril_indices[0], tril_indices[1]] = self.scale_tril.tensor
        tril[range(self.loc.shape[0]), range(self.loc.shape[0])] = tril.diag().exp()
        return tril

    def rsample(self):
        kwargs = {}
        kwargs['scale_tril'] = self.create_tril()
        x = torch.distributions.MultivariateNormal(self.loc.tensor, **kwargs).rsample()
        offset = 0
        if isinstance(self.x, list):
            print(x)
            for xx in self.x:
                xx.tensor = x[offset:(offset + xx.shape[0])]
                print(xx.id, xx.tensor)
                offset += xx.shape[0]
        else:
            self.x.tensor = x
        exit(2)

    def sample(self):
        kwargs = {}
        kwargs['scale_tril'] = self.create_tril()
        x = torch.distributions.MultivariateNormal(self.loc, **kwargs).sample()
        offset = 0
        if isinstance(self.x, list):
            for xx in self.x:
                xx.tensor = x[offset:(offset + xx.shape[0])]
                offset += xx.shape[0]
        else:
            self.x.tensor = x

    def log_prob(self):
        kwargs = {}
        kwargs['scale_tril'] = self.create_tril()
        if isinstance(self.x, list):
            return torch.distributions.MultivariateNormal(self.loc.tensor, **kwargs).log_prob(
                torch.cat([xx.tensor for xx in self.x]))
        else:
            return torch.distributions.MultivariateNormal(self.loc, **kwargs).log_prob(self.x.tensor)

    def update(self, value):
        pass

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        pass

    def __call__(self, *args, **kwargs):
        return self.log_prob()

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        x = process_objects(data['x'], dic)
        loc = process_object(data['parameters']['loc'], dic)
        kwargs = {}
        if 'scale_tril' in data['parameters']:
            kwargs['scale_tril'] = process_object(data['parameters']['scale_tril'], dic)
            if isinstance(x, list):
                x_count = np.sum([xx.shape[0] for xx in x])
            else:
                x_count = x.shape[0]
            assert x_count == loc.shape[0]
            assert (x_count * x_count - x_count) / 2 + x_count == kwargs['scale_tril'].shape[0]
        else:
            raise NotImplementedError('MultivariateNormal is only implemented with scale_tril')

        return cls(id_, x, loc, **kwargs)
