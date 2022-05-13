import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution

from .. import Parameter
from ..core.abstractparameter import AbstractParameter
from ..core.model import CallableModel
from ..core.utils import process_object, register_class
from ..typing import ID
from .tree_model import TimeTreeModel


@register_class
class BDSKModel(CallableModel):
    r"""Birth–death skyline plot as a model for transmission

    Effective population size :math:`R=\frac{\lambda}{\mu + \psi}`

    Total rate of becoming infectious :math:`\delta = \mu + \psi`

    Probability of being sampled :math:`s = \frac{\psi}{\mu + \psi}`

    :param R: effective reproductive number
    :param delta: total rate of becoming non infectious
    :param s: probability of an individual being sampled
    :param rho: probability of an individual being sampled at present
    :param origin: time at which the process starts (i.e. t_0)
    :param times: times of rate shift events
    :param relative_times: times are relative to origin
    :param survival: condition on observing at least one sample
    :param validate_args:
    """

    def __init__(
        self,
        id_: ID,
        tree_model: TimeTreeModel,
        R: AbstractParameter,
        delta: AbstractParameter,
        s: AbstractParameter,
        rho: AbstractParameter,
        origin: AbstractParameter,
        times: AbstractParameter = None,
        relative_times: bool = False,
        survival: bool = True,
    ):
        super().__init__(id_)
        self.tree_model = tree_model
        self.R = R
        self.delta = delta
        self.s = s
        self.rho = rho
        self.origin = origin
        self.times = times
        self.relative_times = relative_times
        self.survival = survival

    def handle_model_changed(self, model, obj, index):
        pass

    @property
    def sample_shape(self) -> torch.Size:
        return max(self.tree_model.node_heights.shape[:-1], self.R.shape[:-1], key=len)

    def _call(self):
        lambda_ = self.R.tensor * self.delta.tensor
        mu = self.delta.tensor - self.s.tensor * self.delta.tensor
        psi = self.s.tensor * self.delta.tensor
        if self.rho.shape[-1] != lambda_.shape[-1]:
            rho = torch.cat(
                (
                    torch.zeros(
                        self.rho.shape[:-1] + (lambda_.shape[-1] - self.rho.shape[-1],)
                    ),
                    self.rho.tensor,
                ),
                -1,
            )
        else:
            rho = self.rho.tensor

        bdsk = PiecewiseConstantBirthDeath(
            lambda_,
            mu,
            psi,
            rho,
            self.origin.tensor,
            times=None if self.times is None else self.times.tensor,
            relative_times=self.relative_times,
            survival=self.survival,
        )
        return bdsk.log_prob(self.tree_model.node_heights)

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        tree = process_object(data[TimeTreeModel.tag], dic)
        R = process_object(data['R'], dic)
        delta = process_object(data['delta'], dic)
        s = process_object(data['s'], dic)
        rho = process_object(data['rho'], dic)
        origin = process_object(data['origin'], dic)

        optionals = {}
        if 'times' in data:
            if isinstance(data['times'], list):
                optionals['times'] = Parameter(None, data['times'])
            else:
                optionals['times'] = process_object(data['times'], dic)
        optionals['survival'] = data.get('survival', True)
        optionals['relative_times'] = data.get('relative_times', False)

        return cls(id_, tree, R, delta, s, rho, origin, **optionals)


class PiecewiseConstantBirthDeath(Distribution):
    r"""Piecewise constant birth death model

    :param lambda_: birth rates
    :param mu: death rates
    :param psi: sampling rates
    :param rho: sampling effort
    :param origin: time at which the process starts (i.e. t_0)
    :param times: times of rate shift events
    :param relative_times: times are relative to origin
    :param survival: condition on observing at least one sample
    :param validate_args:
    """
    arg_constraints = {
        'lambda_': constraints.greater_than_eq(0.0),
        'mu': constraints.positive,
        'psi': constraints.greater_than_eq(0.0),
        'rho': constraints.unit_interval,
    }

    def __init__(
        self,
        lambda_: Tensor,
        mu: Tensor,
        psi: Tensor,
        rho: Tensor,
        origin: Tensor,
        times: Tensor = None,
        relative_times=False,
        survival: bool = True,
        validate_args=None,
    ):
        self.lambda_ = lambda_
        self.mu = mu
        self.psi = psi
        self.rho = rho
        self.times = times
        self.origin = origin
        self.relative_times = relative_times
        self.survival = survival
        batch_shape, event_shape = self.mu.shape[:-1], self.mu.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def log_q(self, A, B, t, t_i):
        """Probability density of lineage alive between time t and t_i gives
        rise to observed clade."""
        e = torch.exp(-A * (t - t_i))
        return torch.log(
            4.0
            * e
            / torch.pow(
                e * (1.0 + B) + (1.0 - B),
                2,
            )
        )

    def log_p(self, t, t_i):
        """Probability density of lineage alive between time t and t_i has no
        descendant at time t_m."""
        sum_term = self.lambda_ + self.mu + self.psi
        m = self.mu.shape[-1]

        A = torch.sqrt(
            torch.pow(self.lambda_ - self.mu - self.psi, 2.0)
            + 4.0 * self.lambda_ * self.psi
        )
        B = torch.zeros_like(self.mu, dtype=self.mu.dtype)
        p = torch.ones(self.mu.shape[:-1] + (m + 1,), dtype=self.mu.dtype)
        exp_A_term = torch.exp(A * (t - t_i))
        inv_2lambda = 1.0 / (2.0 * self.lambda_)

        for i in torch.arange(m - 1, -1, step=-1):
            B[..., i] += (
                (1.0 - 2.0 * (1.0 - self.rho[..., i]) * p[..., i + 1].clone())
                * self.lambda_[..., i]
                + self.mu[..., i]
                + self.psi[..., i]
            ) / A[..., i]
            term = exp_A_term[..., i] * (1.0 + B[..., i])
            one_minus_Bi = 1.0 - B[..., i]
            p[..., i] *= (
                sum_term[..., i]
                - A[..., i] * (term - one_minus_Bi) / (term + one_minus_Bi)
            ) * inv_2lambda[..., i]
        return p, A, B

    def log_prob(self, node_heights: torch.Tensor):
        taxa_shape = node_heights.shape[:-1] + (int((node_heights.shape[-1] + 1) / 2),)
        tip_heights = node_heights[..., : taxa_shape[-1]]
        serially_sampled = torch.any(tip_heights > 0.0)

        m = self.mu.shape[-1]

        if self.times is None:
            dtimes = (self.origin / m).expand(self.origin.shape[:-1] + (m,))
            times = torch.cat(
                (torch.zeros(dtimes.shape[:-1] + (1,), dtype=dtimes.dtype), dtimes), -1
            ).cumsum(-1)
        else:
            times = self.times

        times = torch.broadcast_to(times, self.mu.shape[:-1] + times.shape[-1:])

        if self.relative_times and self.times is not None:
            times = times * self.origin

        p, A, B = self.log_p(times[..., 1:], times[..., :-1])

        # first term
        e = torch.exp(-A[..., 0] * times[..., 1])
        q0 = 4.0 * e / torch.pow(e * (1.0 - B[..., 0]) + (1.0 + B[..., 0]), 2)

        log_p = torch.log(q0)
        # condition on sampling at least one individual
        if self.survival:
            log_p -= torch.log(1.0 - p[..., 0])

        # calculate l(x) with l(t)=1 iff t_{i-1} <= t < t_i
        x = times[..., -1:] - node_heights[..., taxa_shape[-1] :]
        indices_x = torch.max(times.unsqueeze(-2) >= x.unsqueeze(-1), dim=-1)[1] - 1
        log_p += (
            torch.log(self.lambda_.gather(-1, indices_x))
            + self.log_q(
                A.gather(-1, indices_x),
                B.gather(-1, indices_x),
                x,
                torch.gather(times[..., 1:], -1, indices_x),
            )
        ).sum(-1)

        y = times[..., -1:] - tip_heights
        if serially_sampled:
            indices_y = torch.max(times.unsqueeze(-2) >= y.unsqueeze(-1), dim=-1)[1] - 1
            log_p += (
                torch.log(self.psi.gather(-1, indices_y))
                - self.log_q(
                    A.gather(-1, indices_y),
                    B.gather(-1, indices_y),
                    y,
                    times.gather(-1, indices_y + 1),
                )
            ).sum(-1)

        # last term
        if m > 1:
            # number of degree 2 vertices
            ni = (
                torch.sum(x.unsqueeze(-2) < times[..., 1:].unsqueeze(-1), -1)
                - torch.sum(y.unsqueeze(-2) <= times[..., 1:].unsqueeze(-1), -1)
            )[..., :-1] + 1.0

            # contemporenaous term
            log_p += (
                ni
                * (
                    self.log_q(A[..., 1:], B[..., 1:], times[..., 1:-1], times[..., 2:])
                    + torch.log(1.0 - self.rho[..., :-1])
                )
            ).sum(-1)

        N = torch.sum(
            times[..., 1:].unsqueeze(-2)
            == torch.unsqueeze(times[..., -1:] - tip_heights, -1),
            -2,
        )
        mask = (N > 0).logical_and(self.rho > 0.0)
        if torch.any(mask):
            p = torch.masked_select(N, mask) * torch.masked_select(self.rho, mask).log()
            log_p += p.squeeze() if log_p.dim() == 0 else p
        return log_p
