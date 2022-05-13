from __future__ import annotations

import inspect

from torch.optim.lr_scheduler import _LRScheduler as TorchScheduler

from ..core.serializable import JSONSerializable
from ..core.utils import get_class, register_class


@register_class
class Scheduler(JSONSerializable):
    """A wrapper for :class:`~torch.optim.lr_scheduler` objects.

    :param scheduler: a :class:`~torch.optim.lr_scheduler`
    """

    def __init__(self, scheduler: TorchScheduler) -> None:
        self.scheduler = scheduler

    def step(self, *args) -> None:
        self.scheduler.step(*args)

    @classmethod
    def from_json(
        cls, data: dict[str, any], dic: dict[str, any], **kwargs
    ) -> Scheduler:
        scheduler_class = get_class(data['scheduler'])
        signature_params = list(inspect.signature(scheduler_class.__init__).parameters)
        optionals = {}
        for arg in signature_params[1:]:
            if arg in data:
                if arg == 'lr_lambda':
                    optionals['lr_lambda'] = eval(data['lr_lambda'])
                else:
                    optionals[arg] = data[arg]
        scheduler = scheduler_class(kwargs['optimizer'], **optionals)
        return cls(scheduler)
