from __future__ import annotations

import torch

from torchtree.core.abstractparameter import AbstractParameter
from torchtree.core.model import CallableModel
from torchtree.core.utils import process_object, process_objects, register_class
from torchtree.distributions.distributions import DistributionModel
from torchtree.typing import ID


@register_class
class ELBO(CallableModel):
    r"""Class representing the evidence lower bound (ELBO) objective.

    The ELBO is defined as

    .. math::
        \mathcal{L}(q) = \mathbb{E}_q[\log(p(z, x)] - \mathbb{E}_q[\log(q(z; \phi))]

    Maximizing the ELBO wrt variational parameters :math:`\phi` is equivalent
    to minimizing the exclusive Kullback-Leibler divergence from the
    posterior distribution :math:`p` to the variational distribution :math:`q`
    :math:`\text{KL}(q\|p) = \mathbb{E}_q[\log q(z; \phi)]-\mathbb{E}_q[\log p(z| x)]`.

    The shape of ``samples`` is at most 2 dimensional.

    - 0 or 1 dimension N or [N]: standard ELBO.
    - 2 dimensions [N,K]: multi sample ELBO.

    :param id_: ID of ELBO object.
    :type id_: str or None
    :param DistributionModel q: variational distribution.
    :param CallableModel p: joint distribution.
    :param torch.Size samples: number of samples.
    :param bool entropy: use entropy instead of Monte Carlo approximation
        for variational distribution
    :param bool score: use score function instead of pathwise gradient estimator
    """

    def __init__(
        self,
        id_: ID,
        q: DistributionModel,
        p: CallableModel,
        samples: torch.Size,
        entropy=False,
        score=False,
    ) -> None:
        super().__init__(id_)
        self.q = q
        self.p = p
        self.samples = samples
        self.entropy = entropy
        self.score = score

    def _call(self, *args, **kwargs) -> torch.Tensor:
        samples = kwargs.get('samples', self.samples)
        if self.score:
            self.q.sample(samples)
            log_q = self.q()
            with torch.no_grad():
                cost = self.p() - log_q
            lp = (cost * log_q).mean()
        elif len(samples) == 2:
            # Multi sample
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

    def _sample_shape(self) -> torch.Size:
        return self.q.sample_shape

    @classmethod
    def from_json(cls, data, dic) -> ELBO:
        obj = _from_json(cls, data, dic)
        obj.entropy = data.get('entropy', False)
        obj.score = data.get('score', False)
        return obj


@register_class
class KLpq(CallableModel):
    r"""Calculate inclusive Kullback-Leibler divergence from q to p
    :math:`\text{KL}(p\|q)` using self-normalized importance sampling
    gradient estimator.

    The self-normalized importance sampling :footcite:p:`murphy2012machine` estimate
    of :math:`\text{KL}(p \|q)` using the instrument distribution :math:`q` is

    .. math::
        \widehat{KL}(p||q) & = \sum_{s=1}^S \log\left(\frac{p(\tilde{z}_s | x)}
          {q(\tilde{z}_s ; \phi)}\right) w_s , \quad \tilde{z}_s \sim q(z; \phi) \\
        & \propto \sum_{s=1}^S \log\left(\frac{p(\tilde{z}_s)}
          {q(\tilde{z}_s;\phi)}\right) w_s

    where

    .. math::
        w_s = \frac{p(\tilde{z}_s, D, \tau)}{ q(\tilde{z}_s; \phi)} /
          \sum_{i=1}^N \frac{p(\tilde{z}_i, D, \tau)}{q(\tilde{z}_i; \phi)}.


    :param id_: ID of KLpq object.
    :type id_: str or None
    :param DistributionModel q: variational distribution.
    :param CallableModel p: joint distribution.
    :param torch.Size samples: number of samples.

    .. footbibliography::
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

    def _sample_shape(self) -> torch.Size:
        return self.q.sample_shape

    @classmethod
    def from_json(cls, data, dic) -> KLpq:
        return _from_json(cls, data, dic)


@register_class
class KLpqImportance(CallableModel):
    r"""Class for minimizing inclusive Kullback-Leibler divergence
    from q to p :math:`\text{KL}(p\|q)` using self-normalized importance
    sampling gradient estimator.

    .. math::
        \nabla \widehat{\text{KL}}(p\|q) = -\sum_{s=1}^S w_s
          \nabla\log q(\tilde{z}_s ; \phi) , \quad \tilde{z}_s \sim q(z; \phi)

    where

    .. math::
        w_s = \frac{p(\tilde{z}_s, D, \tau)}{ q(\tilde{z}_s; \phi)} /
          \sum_{i=1}^N \frac{p(\tilde{z}_i, D, \tau)}{q(\tilde{z}_i; \phi)}.

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
        with torch.no_grad():
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

    def _sample_shape(self) -> torch.Size:
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
    r"""Class representing the stratified evidence lower bound (SELBO) objective.

    Maximizing the SELBO is equivalent to minimizing exclusive Kullback-Leibler
    divergence from p to q :math:`\text{KL}(q\|p)` where :math:`q=\sum_i \alpha_i q_i`
    :footcite:p:`morningstar2021automatic`.

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

    .. footbibliography::
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

    def _sample_shape(self) -> torch.Size:
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
