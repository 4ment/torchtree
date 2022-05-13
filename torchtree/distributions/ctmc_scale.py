from typing import Optional, Union

import torch

from ..core.abstractparameter import AbstractParameter
from ..core.model import CallableModel
from ..core.utils import process_object, register_class
from ..evolution.tree_model import TreeModel
from ..typing import ID


# Code adapted from
# github.com/beast-dev/beast-mcmc/blob/master/src/dr/evomodel/tree/CTMCScalePrior.java
@register_class
class CTMCScale(CallableModel):
    """Class implementing the CTMC scale prior [#ferreira2008]_

    :param id_: ID of object
    :type id_: str or None
    :param torch.Tensor x: substitutin rate parameter
    :param TreeModel tree_model: tree model

    .. [#ferreira2008] Ferreira and Suchard. Bayesian analysis of elapsed times
     in continuous-time Markov chains. 2008
    """

    shape = torch.tensor([0.5])
    log_gamma_one_half = torch.lgamma(shape)

    def __init__(self, id_: ID, x: AbstractParameter, tree_model: TreeModel) -> None:
        super(CTMCScale, self).__init__(id_)
        self.x = x
        self.tree_model = tree_model

    def _call(self, *args, **kwargs) -> torch.Tensor:
        total_tree_time = self.tree_model.branch_lengths().sum(-1, keepdim=True)
        log_normalization = (
            self.shape * torch.log(total_tree_time) - self.log_gamma_one_half
        )
        log_like = (
            log_normalization
            - self.shape * self.x.tensor.log()
            - self.x.tensor * total_tree_time
        )
        return log_like

    @property
    def sample_shape(self) -> torch.Size:
        return self.x.tensor.shape[:-1]

    def to(self, *args, **kwargs) -> None:
        super().to(*args, **kwargs)
        self.shape = self.shape.to(*args, **kwargs)
        self.log_gamma_one_half = self.log_gamma_one_half.to(*args, **kwargs)

    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> None:
        super().cuda(device)
        self.shape = self.shape.cuda(device)
        self.log_gamma_one_half = self.log_gamma_one_half.cuda(device)

    def cpu(self) -> None:
        super().cpu()
        self.shape = self.shape.cpu()
        self.log_gamma_one_half = self.log_gamma_one_half.cpu()

    @staticmethod
    def json_factory(id_: str, rate: Union[str, dict], tree: Union[str, dict]) -> dict:
        return {
            'id': id_,
            'type': 'CTMCScale',
            'x': rate,
            'tree_model': tree,
        }

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        x = process_object(data['x'], dic)
        tree_model = process_object(data['tree_model'], dic)
        return cls(id_, x, tree_model)
