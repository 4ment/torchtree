import torch

from ..core.model import CallableModel
from ..core.utils import process_object
from ..distributions.distributions import DistributionModel
from ..typing import ID


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
    """

    def __init__(
        self, id_: ID, q: DistributionModel, p: CallableModel, samples: torch.Size
    ) -> None:
        super().__init__(id_)
        self.q = q
        self.p = p
        self.samples = samples
        self.add_model(q)
        self.add_model(p)

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
            lp = (self.p() - self.q()).mean()
        return lp

    def update(self, value):
        pass

    def handle_model_changed(self, model, obj, index):
        self.fire_model_changed()

    def handle_parameter_changed(self, variable, index, event):
        pass

    @property
    def sample_shape(self) -> torch.Size:
        return self.q.sample_shape

    @classmethod
    def from_json(cls, data, dic) -> 'ELBO':
        return _from_json(cls, data, dic)


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
        self.add_model(q)
        self.add_model(p)

    def _call(self, *args, **kwargs) -> torch.Tensor:
        samples = kwargs.get('samples', self.samples)
        self.q.sample(samples)
        log_w = self.p() - self.q()
        log_w_norm = log_w - torch.logsumexp(log_w, -1)
        return torch.sum(log_w_norm.exp() * log_w)

    def update(self, value):
        pass

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        pass

    @property
    def sample_shape(self) -> torch.Size:
        return self.q.sample_shape

    @classmethod
    def from_json(cls, data, dic) -> 'KLpq':
        return _from_json(cls, data, dic)


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
        self.add_model(q)
        self.add_model(p)

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

    def update(self, value):
        pass

    def handle_model_changed(self, model, obj, index):
        pass

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
