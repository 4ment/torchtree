import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution

from ..core.abstractparameter import AbstractParameter
from ..core.model import CallableModel
from ..core.utils import process_object, register_class
from ..typing import ID
from .tree_model import TimeTreeModel


@register_class
class BirthDeathModel(CallableModel):
    r"""Birth–death model

    :param lambda_: birth rate
    :param mu: death rate
    :param psi: sampling rate
    :param rho: sampling effort
    :param origin: time at which the process starts (i.e. t_0)
    :param survival: condition on observing at least one sample
    """

    def __init__(
        self,
        id_: ID,
        tree_model: TimeTreeModel,
        lambda_: AbstractParameter,
        mu: AbstractParameter,
        psi: AbstractParameter,
        rho: AbstractParameter,
        origin: AbstractParameter,
        survival: bool = True,
    ):
        super().__init__(id_)
        self.tree_model = tree_model
        self.lambda_ = lambda_
        self.mu = mu
        self.psi = psi
        self.rho = rho
        self.origin = origin
        self.survival = survival

    def handle_model_changed(self, model, obj, index):
        pass

    @property
    def sample_shape(self) -> torch.Size:
        return max(
            self.tree_model.node_heights.shape[:-1], self.lambda_.shape[:-1], key=len
        )

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

        bd = BirthDeath(
            lambda_,
            mu,
            psi,
            rho,
            self.origin.tensor,
            survival=self.survival,
        )
        return bd.log_prob(self.tree_model.node_heights)

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        tree = process_object(data[TimeTreeModel.tag], dic)
        R = process_object(data['lambda'], dic)
        delta = process_object(data['mu'], dic)
        s = process_object(data['psi'], dic)
        rho = process_object(data['rho'], dic)
        origin = process_object(data['origin'], dic)

        optionals = {}
        optionals['survival'] = data.get('survival', True)

        return cls(id_, tree, R, delta, s, rho, origin, **optionals)


class BirthDeath(Distribution):
    r"""Constant birth death model

    :param lambda_: birth rate
    :param mu: death rate
    :param psi: sampling rate
    :param rho: sampling effort
    :param origin: time at which the process starts (i.e. t_0)
    :param survival: condition on observing at least one sample
    :param validate_args:
    """
    arg_constraints = {
        'lambda_': constraints.greater_than_eq(0.0),
        'mu': constraints.positive,
        'psi': constraints.greater_than_eq(0.0),
        'rho': constraints.unit_interval,
        'origin': constraints.positive,
    }

    def __init__(
        self,
        lambda_: Tensor,
        mu: Tensor,
        psi: Tensor,
        rho: Tensor,
        origin: Tensor,
        survival: bool = True,
        validate_args=None,
    ):
        self.lambda_ = lambda_
        self.mu = mu
        self.psi = psi
        self.rho = rho
        self.origin = origin
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

    def log_p(self, t):
        """Probability density of lineage alive between time t and t_i has no
        descendant at time t_m."""
        A = torch.sqrt(
            torch.pow(self.lambda_ - self.mu - self.psi, 2.0)
            + 4.0 * self.lambda_ * self.psi
        )
        B = ((1.0 - 2.0 * (1.0 - self.rho)) * self.lambda_ + self.mu + self.psi) / A
        term = torch.exp(A * t) * (1.0 + B)
        one_minus_Bi = 1.0 - B
        p = (
            self.lambda_
            + self.mu
            + self.psi
            - A * (term - one_minus_Bi) / (term + one_minus_Bi)
        ) / (2.0 * self.lambda_)
        return p, A, B

    def log_prob(self, node_heights: torch.Tensor):
        taxa_shape = node_heights.shape[:-1] + (int((node_heights.shape[-1] + 1) / 2),)
        tip_heights = node_heights[..., : taxa_shape[-1]]
        serially_sampled = torch.any(tip_heights > 0.0)

        p, A, B = self.log_p(self.origin)

        # first term
        e = torch.exp(-A * self.origin)
        q0 = 4.0 * e / torch.pow(e * (1.0 - B) + (1.0 + B), 2)

        log_p = torch.log(q0)
        # condition on sampling at least one individual
        if self.survival:
            log_p -= torch.log(1.0 - p[..., 0])

        # calculate l(x) with l(t)=1 iff t_{i-1} <= t < t_i
        x = self.origin - node_heights[..., taxa_shape[-1] :]
        log_p += (
            torch.log(self.lambda_)
            + self.log_q(
                A,
                B,
                x,
                self.origin,
            )
        ).sum(-1)

        y = self.origin - tip_heights
        if serially_sampled:
            log_p += (
                torch.log(self.psi)
                - self.log_q(
                    A,
                    B,
                    y,
                    self.origin,
                )
            ).sum(-1)
        return log_p
