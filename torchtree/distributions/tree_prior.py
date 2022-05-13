import torch
from torch import Tensor

from ..core.abstractparameter import AbstractParameter
from ..core.model import CallableModel
from ..core.parameter import Parameter
from ..core.utils import process_object, register_class
from ..evolution.tree_model import UnRootedTreeModel
from ..typing import ID


@register_class
class CompoundGammaDirichletPrior(CallableModel):
    """Compound gamma-Dirichlet prior on an unrooted tree [rannala2011]_

    :param id_: ID of object
    :param UnRootedTreeModel tree_model: unrooted tree model
    :param AbstractParameter alpha: concentration parameter of Dirichlet distribution
    :param AbstractParameter c: ratio of the mean internal/external branch lengths
    :param AbstractParameter shape: shape parameter of the gamma distribution
    :param AbstractParameter rate: rate parameter of the gamma distribution

    .. [rannala2011]  Rannala, Zhu, and Ziheng Yang. Tail Paradox, Partial
     Identifiability, and Influential Priors in Bayesian Branch Length Inference. 2011
    """

    def __init__(
        self,
        id_: ID,
        tree_model: UnRootedTreeModel,
        alpha: AbstractParameter,
        c: AbstractParameter,
        shape: AbstractParameter,
        rate: AbstractParameter,
    ):
        super().__init__(id_)
        self.tree_model = tree_model
        self.alpha = alpha
        self.c = c
        self.shape = shape
        self.rate = rate

    def _call(self, *args, **kwargs) -> Tensor:
        taxa_count = self.tree_model.taxa_count
        x = self.tree_model.branch_lengths()
        sum_x = x.sum(-1)
        return (
            torch.sum(x[..., :taxa_count].log(), -1) * (self.alpha.tensor - 1)
            + torch.sum(x[..., taxa_count:].log(), -1)
            * (self.c.tensor * self.alpha.tensor - 1)
            - torch.lgamma(self.alpha.tensor) * taxa_count
            - torch.lgamma(self.c.tensor * self.alpha.tensor) * (taxa_count - 3)
            + torch.lgamma(
                self.alpha.tensor * taxa_count
                + (taxa_count - 3) * self.c.tensor * self.alpha.tensor
            )
            + sum_x.log()
            * (
                self.shape.tensor
                - self.alpha.tensor * taxa_count
                - self.alpha.tensor * self.c.tensor * (taxa_count - 3)
            )
            + self.shape.tensor * torch.log(self.rate.tensor)
            - torch.lgamma(self.shape.tensor)
            - self.rate.tensor * sum_x
        )

    @property
    def sample_shape(self) -> torch.Size:
        return self.tree_model.sample_shape

    def handle_parameter_changed(self, variable, index, event) -> None:
        pass

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        tree_model = process_object(data['tree_model'], dic)
        if isinstance(data['alpha'], (float, int)):
            alpha = Parameter(None, torch.tensor([float(data['alpha'])]))
        else:
            alpha = process_object(data['alpha'], dic)

        if isinstance(data['c'], (float, int)):
            c = Parameter(None, torch.tensor([float(data['c'])]))
        else:
            c = process_object(data['c'], dic)

        if isinstance(data['shape'], (float, int)):
            shape = Parameter(None, torch.tensor([float(data['shape'])]))
        else:
            shape = process_object(data['shape'], dic)

        if isinstance(data['rate'], (float, int)):
            rate = Parameter(None, torch.tensor([float(data['rate'])]))
        else:
            rate = process_object(data['rate'], dic)
        return cls(id_, tree_model, alpha, c, shape, rate)
