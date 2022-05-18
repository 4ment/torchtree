"""Masked Autoregressive Flow for Density Estimation arXiv:1705.07057v4 Code
ported from https://github.com/kamenbliznashki/normalizing_flows."""
from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from .. import Parameter
from ..core.abstractparameter import AbstractParameter
from ..core.container import Container
from ..core.utils import process_object, process_objects, register_class
from ..distributions.distributions import Distribution, DistributionModel


class LinearMaskedCoupling(nn.Module):
    """Modified RealNVP Coupling Layers per the MAF paper."""

    def __init__(self, input_size, hidden_size, n_hidden, mask, cond_label_size=None):
        super().__init__()

        self.register_buffer('mask', mask)

        # scale function
        s_net = [
            nn.Linear(
                input_size + (cond_label_size if cond_label_size is not None else 0),
                hidden_size,
            )
        ]
        for _ in range(n_hidden):
            s_net += [nn.Tanh(), nn.Linear(hidden_size, hidden_size)]
        s_net += [nn.Tanh(), nn.Linear(hidden_size, input_size)]
        self.s_net = nn.Sequential(*s_net)

        # translation function
        self.t_net = copy.deepcopy(self.s_net)
        # replace Tanh with ReLU's per MAF paper
        for i in range(len(self.t_net)):
            if not isinstance(self.t_net[i], nn.Linear):
                self.t_net[i] = nn.ReLU()

    def forward(self, x, y=None):
        # apply mask
        mx = x * self.mask
        # run through model
        s = self.s_net(mx if y is None else torch.cat([y, mx], dim=1))
        t = self.t_net(mx if y is None else torch.cat([y, mx], dim=1))
        u = mx + (1 - self.mask) * (x - t) * torch.exp(
            -s
        )  # cf RealNVP eq 8 where u corresponds to x (here we're modeling u)

        log_abs_det_jacobian = -(1 - self.mask) * s
        # log det du/dx; cf RealNVP 8 and 6; note, sum over input_size
        # done at model log_prob

        return u, log_abs_det_jacobian

    def inverse(self, u, y=None):
        # apply mask
        mu = u * self.mask

        # run through model
        s = self.s_net(mu if y is None else torch.cat([y, mu], dim=1))
        t = self.t_net(mu if y is None else torch.cat([y, mu], dim=1))
        x = mu + (1 - self.mask) * (u * s.exp() + t)  # cf RealNVP eq 7

        log_abs_det_jacobian = (1 - self.mask) * s  # log det dx/du

        return x, log_abs_det_jacobian


class BatchNorm(nn.Module):
    """RealNVP BatchNorm layer."""

    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.ones(input_size))

    def forward(self, x, cond_y=None):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = x.var(0)
            # note MAF paper uses biased variance estimate; ie x.var(0, unbiased=False)

            # update running mean
            self.running_mean.mul_(self.momentum).add_(
                self.batch_mean.data * (1 - self.momentum)
            )
            self.running_var.mul_(self.momentum).add_(
                self.batch_var.data * (1 - self.momentum)
            )

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # compute normalized input (cf original batch norm paper algo 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta

        # compute log_abs_det_jacobian (cf RealNVP paper)
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)
        return y, log_abs_det_jacobian.expand_as(x)

    def inverse(self, y, cond_y=None):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma

        return x, log_abs_det_jacobian.expand_as(x)


class FlowSequential(nn.Sequential):
    """Container for layers of a normalizing flow."""

    def forward(self, x, y):
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians

    def inverse(self, u, y):
        sum_log_abs_det_jacobians = 0
        for module in reversed(self):
            u, log_abs_det_jacobian = module.inverse(u, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return u, sum_log_abs_det_jacobians


@register_class
class RealNVP(DistributionModel):
    """Class for RealNVP normalizing flows.

    :param id_: ID of object
    :param x: parameter or list of parameters
    :param base: base distribution
    :param n_blocks:
    :param hidden_size:
    :param n_hidden:
    :param cond_label_size:
    :param batch_norm:
    """

    def __init__(
        self,
        id_: str,
        x: AbstractParameter,
        base: Distribution,
        n_blocks: int,
        hidden_size: int,
        n_hidden: int,
        cond_label_size=None,
        batch_norm=False,
    ) -> None:
        DistributionModel.__init__(self, id_)
        self.x = x
        self.base_dist = base
        base.remove_model_listener(self)
        self.sum_log_abs_det_jacobians = None
        self.input_size = base.batch_shape[0]

        # construct model
        modules = []
        mask = torch.arange(self.input_size).float() % 2
        for i in range(n_blocks):
            modules += [
                LinearMaskedCoupling(
                    self.input_size, hidden_size, n_hidden, mask, cond_label_size
                )
            ]
            mask = 1 - mask
            modules += batch_norm * [BatchNorm(self.input_size)]

        self.net = FlowSequential(*modules)
        tensors = []
        for idx, tensor in enumerate(self.net.parameters()):
            tensors.append(Parameter(f"{id_}.realnvp.{idx}", tensor))
        self.net_parameters = Container(None, tensors)

    def forward(self, x, y=None):
        return self.net(x, y)

    def inverse(self, u, y=None):
        return self.net.inverse(u, y)

    def apply_flow(self, sample_shape: torch.Size):
        if sample_shape == torch.Size([]):
            zz, self.sum_log_abs_det_jacobians = self.forward(
                self.base_dist.x.tensor.unsqueeze(0)
            )
            zz = zz.squeeze()
        else:
            zz, self.sum_log_abs_det_jacobians = self.forward(self.base_dist.x.tensor)

        if isinstance(self.x, (list, tuple)):
            offset = 0
            for xx in self.x:
                xx.tensor = zz[..., offset : (offset + xx.shape[-1])]
                offset += xx.shape[-1]
        else:
            self.x.tensor = zz

    def sample(self, sample_shape=torch.Size()) -> None:
        self.base_dist.sample(sample_shape)
        self.apply_flow(sample_shape)

    def rsample(self, sample_shape=torch.Size()) -> None:
        self.base_dist.rsample(sample_shape)
        self.apply_flow(sample_shape)

    def log_prob(self, x: AbstractParameter = None) -> Tensor:
        # negative log_det because sum_log_abs_det_jacobians is coming from forward
        # transformation and we need the det of backward transformation
        return torch.sum(self.base_dist() - self.sum_log_abs_det_jacobians, dim=-1)

    def _call(self, *args, **kwargs) -> Tensor:
        return self.log_prob()

    @property
    def batch_shape(self) -> torch.Size:
        return self.base_dist.batch_shape

    @property
    def sample_shape(self) -> torch.Size:
        return self.base.sample_shape

    def parameters(self) -> list[AbstractParameter]:
        return self.net_parameters.parameters()

    def entropy(self) -> torch.Tensor:
        pass

    @classmethod
    def from_json(cls, data, dic) -> RealNVP:
        id_ = data['id']
        x = process_objects(data['x'], dic)
        base = process_object(data['base'], dic)
        n_blocks = data['n_blocks']
        hidden_size = data['hidden_size']
        n_hidden = data['n_hidden']

        real_nvp = cls(id_, x, base, n_blocks, hidden_size, n_hidden)
        if 'state' in data:
            real_nvp.net.load_state_dict(torch.load(data['state']))
        return cls(id_, x, base, n_blocks, hidden_size, n_hidden)
