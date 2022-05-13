import torch
import torch.distributions.normal

from ..core.abstractparameter import AbstractParameter
from ..core.model import CallableModel
from ..core.parameter import Parameter
from ..core.utils import process_object, register_class
from ..evolution.tree_model import TimeTreeModel, TreeModel
from ..typing import ID


@register_class
class GMRF(CallableModel):
    def __init__(
        self,
        id_: ID,
        field: AbstractParameter,
        precision: AbstractParameter,
        tree_model: TimeTreeModel = None,
        weights: torch.Tensor = None,
        rescale: bool = True,
    ) -> None:
        super().__init__(id_)
        self.tree_model = tree_model
        self.weights = weights
        self.field = field
        self.precision = precision
        self.rescale = rescale

    def _call(self, *args, **kwargs) -> torch.Tensor:
        diff_square = torch.pow(
            self.field.tensor[..., :-1] - self.field.tensor[..., 1:], 2.0
        )
        if self.tree_model is not None:
            heights = torch.cat(
                (
                    torch.zeros(
                        self.tree_model.node_heights.shape[:-1] + (1,),
                        dtype=self.field.dtype,
                        device=self.field.device,
                    ),
                    self.tree_model.node_heights[..., self.tree_model.taxa_count :],
                ),
                -1,
            )
            indices = torch.argsort(heights, descending=False)
            heights_sorted = torch.gather(heights, -1, indices)
            durations = heights_sorted[..., 1:] - heights_sorted[..., :-1]
            diff_square /= (durations[..., :-1] + durations[..., 1:]) / 2.0
            if self.rescale:
                diff_square *= heights_sorted[..., -1:]
        elif self.weights is not None:
            diff_square /= self.weights

        dim = self.field.shape[-1] - 1.0  # field dim
        precision = self.precision.tensor
        return (
            precision.log() * dim / 2.0
            - diff_square.sum(-1, keepdim=True) * precision / 2.0
            - dim / 2.0 * 1.8378770664093453
        )

    @property
    def sample_shape(self) -> torch.Size:
        return self.field.tensor.shape[:-1]

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        tree_model = None
        weights = None
        # time-aware if a tree_model is provided
        if TreeModel.tag in data:
            tree_model = process_object(data[TreeModel.tag], dic)
        elif 'weights' in data:
            weights = process_object(data['weights'], dic)
        field = process_object(data['x'], dic)
        precision = process_object(data['precision'], dic)
        rescale = data.get('rescale', True)
        return cls(id_, field, precision, tree_model, weights, rescale)


@register_class
class GMRFCovariate(CallableModel):
    def __init__(
        self,
        id_: ID,
        field: AbstractParameter,
        precision: AbstractParameter,
        covariates: AbstractParameter,
        beta: AbstractParameter,
    ) -> None:
        super().__init__(id_)
        self.field = field
        self.precision = precision
        self.covariates = covariates
        self.beta = beta

    def _call(self, *args, **kwargs) -> torch.Tensor:
        dim = self.field.shape[-1]
        precision = self.precision.tensor
        covariates = (
            self.covariates.tensor
            if self.covariates.shape[:-2] == self.beta.shape[:-1]
            else self.covariates.tensor.expand(self.beta.shape[:-1], (-1,))
        )
        design_matrix = torch.zeros(
            self.field.shape[:-1] + (dim, dim),
            dtype=self.field.dtype,
            device=self.field.device,
        )
        design_matrix[..., range(dim - 1), range(1, dim)] = design_matrix[
            ..., range(1, dim), range(dim - 1)
        ] = -precision.expand(self.field.shape[:-1] + (dim - 1,))

        design_matrix[..., range(1, dim - 1), range(1, dim - 1)] = 2.0 * precision
        design_matrix[..., 0, 0] = design_matrix[..., dim - 1, dim - 1] = precision
        field_z_beta = self.field.tensor - (covariates @ self.beta.tensor)
        return (
            0.5 * (dim - 1) * precision.log()
            - 0.5 * field_z_beta.t() @ design_matrix @ field_z_beta
            - (dim - 1) / 2.0 * 1.8378770664093453
        )

    @property
    def sample_shape(self) -> torch.Size:
        return self.field.tensor.shape[:-1]

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        field = process_object(data['field'], dic)
        precision = process_object(data['precision'], dic)
        if isinstance(data['covariates'], list):
            covariates = Parameter(
                None,
                torch.tensor(
                    data['covariates'], dtype=field.dtype, device=field.device
                ),
            )
        else:
            covariates = process_object(data['covariates'], dic)
        beta = process_object(data['beta'], dic)
        return cls(id_, field, precision, covariates, beta)
