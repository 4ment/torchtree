import torch
from torch import Tensor, nn
from torch.nn import Parameter

from torchtree.core.utils import register_class


@register_class
class PlanarTransform(nn.Module):
    r"""
    Implementation of the transformation used in planar flow:

    f(z) = z + u * tanh(dot(w.T, z) + b)

    where z are the inputs and u, w, and b are learnable parameters.
    The shape of z is (batch_size, input_size).

    :param Parameter u: scaling factor with shape(1, input_size)
    :param Parameter w: weight with shape (1, input_size)
    :param Parameter b: bias with shape (1)
    """

    def __init__(self, u: Parameter, w: Parameter, b: Parameter) -> None:
        super().__init__()
        self.u = u  # (1, input_size)
        self.w = w  # (1, input_size)
        self.b = b  # (1)
        self._log_det_jacobian = None

    def forward(self, z: Tensor) -> Tensor:
        u_hat = self.u_hat()
        a = torch.tanh(z @ self.w.t() + self.b)

        # calculate log det jacobian
        psi = (1 - a**2) @ self.w
        self._log_det_jacobian = torch.log(
            torch.abs(1 + torch.matmul(psi, u_hat.t()))
        ).squeeze()

        return z + u_hat * a

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return self._log_det_jacobian

    #  Enforce w^T u >= -1: sufficient condition for invertibility
    def u_hat(self) -> Tensor:
        alpha = (self.w @ self.u.t()).squeeze()
        if alpha >= -1:
            return self.u
        a_prime = -1 + torch.log1p(alpha.exp())
        return self.u + (a_prime - alpha) * self.w / (self.w @ self.w.T)
