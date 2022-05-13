from __future__ import annotations

import torch

from ..core.abstractparameter import AbstractParameter
from ..core.model import CallableModel
from ..core.utils import process_object, process_objects, register_class
from ..distributions.distributions import DistributionModel
from ..typing import ID


@register_class
class ELBO(CallableModel):
    r"""
    Class representing the evidence lower bound (ELBO) objective.
    Maximizing the ELBO is equivalent to minimizing exclusive Kullback-Leibler
    divergence from p to q :math:`KL(q\|p)`.

    The shape of ``samples`` is at most 2 dimensional.

    - 0 or 1 dimension N or [N]: standard ELBO.
    - 2 dimensions [N,K]: multi sample ELBO.

    :param id_: ID of KLqp object.
    :type id_: str or None
    :param DistributionModel q: variational distribution.
    :param CallableModel p: joint distribution.
    :param torch.Size samples: number of samples.
    :param bool entropy: use entropy instead of Monte Carlo approximation
    for variational distribution
    """

    def __init__(
        self,
        id_: ID,
        q: DistributionModel,
        p: CallableModel,
        samples: torch.Size,
        entropy=False,
    ) -> None:
        super().__init__(id_)
        self.q = q
        self.p = p
        self.samples = samples
        self.entropy = entropy

    def _call(self, *args, **kwargs) -> torch.Tensor:
        samples = kwargs.get('samples', self.samples)
        # Multi sample
        if len(samples) == 2:
            self.q.rsample(samples)
            log_q = self.q()
            log_p = self.p()
            lp = (
                torch.logsumexp(log_p - log_q, -1)
                - torch.tensor(float(log_p.shape[-1])).log()
            ).mean()
        else:
            self.q.rsample(samples)

            if self.entropy:
                lp = self.p().mean() + self.q.entropy().sum()
            else:
                lp = (self.p() - self.q()).mean()
        return lp

    def handle_parameter_changed(self, variable, index, event):
        pass

    @property
    def sample_shape(self) -> torch.Size:
        return self.q.sample_shape

    @classmethod
    def from_json(cls, data, dic) -> ELBO:
        obj = _from_json(cls, data, dic)
        obj.entropy = data.get('entropy', False)
        return obj


@register_class
class KLpq(CallableModel):
    r"""
    Calculate inclusive Kullback-Leibler divergence from q to p :math:`KL(p\|q)`
    using self-normalized importance sampling gradient estimator [#oh1992]_.

    :param id_: ID of KLpq object.
    :type id_: str or None
    :param DistributionModel q: variational distribution.
    :param CallableModel p: joint distribution.
    :param torch.Size samples: number of samples.

    .. [#oh1992] Oh, M.-S., & Berger, J. O. (1992). Adaptive importance sampling in
     Monte Carlo integration.
     Journal of Statistical Computation and Simulation, 41(3-4), 143–168.
    """

    def __init__(
        self, id_: ID, q: DistributionModel, p: CallableModel, samples: torch.Size
    ) -> None:
        super().__init__(id_)
        self.q = q
        self.p = p
        self.samples = samples

    def _call(self, *args, **kwargs) -> torch.Tensor:
        samples = kwargs.get('samples', self.samples)
        self.q.sample(samples)
        log_w = self.p() - self.q()
        log_w_norm = log_w - torch.logsumexp(log_w, -1)
        return torch.sum(log_w_norm.exp() * log_w)

    def handle_parameter_changed(self, variable, index, event):
        pass

    @property
    def sample_shape(self) -> torch.Size:
        return self.q.sample_shape

    @classmethod
    def from_json(cls, data, dic) -> KLpq:
        return _from_json(cls, data, dic)


@register_class
class KLpqImportance(CallableModel):
    r"""
    Class for minimizing inclusive Kullback-Leibler divergence
    from q to p :math:`KL(p\|q)`
    using self-normalized importance sampling gradient estimator [#oh1992]_.

    :param id_: ID of object.
    :type id_: str or None
    :param DistributionModel q: variational distribution.
    :param CallableModel p: joint distribution.
    :param torch.Size samples: number of samples.

    """

    def __init__(
        self, id_: ID, q: DistributionModel, p: CallableModel, samples: torch.Size
    ) -> None:
        super().__init__(id_)
        self.q = q
        self.p = p
        self.samples = samples

    def _call(self, *args, **kwargs) -> torch.Tensor:
        samples = kwargs.get('samples', self.samples)
        self.q.sample(samples)
        log_p = self.p()
        log_q = self.q()
        log_w = log_p - log_q.detach()
        w = torch.exp(log_w - log_w.max())
        w_norm = w / w.sum()
        return -torch.sum(w_norm * log_q)
        # log_w_norm = log_w - torch.logsumexp(log_w, -1)
        # return torch.sum(log_w_norm.exp() * log_q)

    def handle_parameter_changed(self, variable, index, event):
        pass

    @property
    def sample_shape(self) -> torch.Size:
        return self.q.sample_shape

    @classmethod
    def from_json(cls, data, dic):
        return _from_json(cls, data, dic)


def _from_json(cls, data, dic):
    samples = data.get('samples', 1)
    if isinstance(samples, list):
        samples = torch.Size(samples)
    else:
        samples = torch.Size((samples,))

    var_desc = data['variational']
    var = process_object(var_desc, dic)

    joint_desc = data['joint']
    joint = process_object(joint_desc, dic)
    return cls(data['id'], var, joint, samples)


@register_class
class SELBO(CallableModel):
    r"""
    Class representing the stratified evidence lower bound (SELBO) objective.
    Maximizing the SELBO is equivalent to minimizing exclusive Kullback-Leibler
    divergence from p to q :math:`KL(q\|p)` where :math:`q=\sum_i \alpha_i q_i`.

    The shape of ``samples`` is at most 2 dimensional.

    - 0 or 1 dimension N or [N]: standard ELBO.
    - 2 dimensions [N,K]: multi sample ELBO.

    :param id_: ID of KLqp object.
    :type id_: str or None
    :param DistributionModel components: list of distribution.
    :param AbstractParameter weights:
    :param CallableModel p: joint distribution.
    :param torch.Size samples: number of samples.
    :param bool entropy: use entropy instead of Monte Carlo approximation
    for variational distribution
    """

    def __init__(
        self,
        id_: ID,
        components: list[DistributionModel],
        weights: AbstractParameter,
        p: CallableModel,
        samples: torch.Size,
        entropy=False,
    ) -> None:
        super().__init__(id_)
        self.components = components
        self.p = p
        self.weights = weights
        self.samples = samples
        self.entropy = entropy

    def _call(self, *args, **kwargs) -> torch.Tensor:
        samples = kwargs.get('samples', self.samples)
        # Multi sample
        if len(samples) == 2:
            log_q = []
            for q in self.components:
                q.rsample(samples)
                log_q.append(q())
            log_q = torch.cat(log_q)
            log_p = self.p()
            log_weights = self.weights.tensor.log()
            lp = (
                torch.logsumexp(log_p - log_q + log_weights, -1)
                - torch.tensor(float(log_p.shape[-1])).log()
            ).mean()
        else:
            log_probs = []

            for q in self.components:
                q.rsample(samples)

                if self.entropy:
                    log_probs.append(self.p().mean() + q.entropy())
                else:
                    log_probs.append((self.p() - q().sum(-1)).mean().unsqueeze(0))
            lp = (self.weights.tensor * torch.cat(log_probs)).sum()
        return lp

    def handle_parameter_changed(self, variable, index, event):
        pass

    @property
    def sample_shape(self) -> torch.Size:
        return self.q.sample_shape

    @classmethod
    def from_json(cls, data, dic):
        samples = data.get('samples', 1)
        if isinstance(samples, list):
            samples = torch.Size(samples)
        else:
            samples = torch.Size((samples,))

        components = process_objects(data['components'], dic)
        weights = process_object(data['weights'], dic)
        joint = process_object(data['joint'], dic)
        print(components)
        return cls(data['id'], components, weights, joint, samples)
