import torch.distributions
import torch.distributions.constraints


class OneOnX(torch.distributions.Distribution):
    arg_constraints = {}
    support = torch.distributions.constraints.positive

    def __init__(self, validate_args=None) -> None:
        super().__init__(torch.Size(), validate_args=validate_args)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)
        return -value.log()
