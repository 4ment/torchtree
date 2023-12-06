from __future__ import annotations

from torchtree.core.identifiable import Identifiable
from torchtree.core.logger import LoggerInterface
from torchtree.core.runnable import Runnable
from torchtree.core.utils import process_object, process_objects, register_class
from torchtree.distributions.distributions import DistributionModel
from torchtree.typing import ID


@register_class
class Sampler(Identifiable, Runnable):
    r"""Class for sampling a distribution and optionally logging things.

    :param DistributionModel model: model to sample from.
    :param int samples: number of sample to draw.
    :param loggers: list of loggers.
    :type loggers: list[LoggerInterface]
    """

    def __init__(
        self,
        id_: ID,
        model: DistributionModel,
        samples: int,
        loggers: list[LoggerInterface],
    ) -> None:
        Identifiable.__init__(self, id_)
        self.model = model
        self.samples = samples
        self.loggers = loggers

    def run(self) -> None:
        """Run sampler: sample and log to loggers."""
        for logger in self.loggers:
            logger.initialize()
        for _ in range(self.samples):
            self.model.sample()
            for logger in self.loggers:
                logger.log()
        for logger in self.loggers:
            logger.close()

    @classmethod
    def from_json(cls, data, dic) -> Sampler:
        r"""Create a Sampler object.

        :param dict[str, Any] data: dictionary representation of a Sampler object.
        :param dict[str, Identifiable] dic: dictionary containing torchtree objects
            keyed by their ID.
        :return: a :class:`~torchtree.inference.sampler.Sampler` object.
        """
        model = process_object(data['model'], dic)
        loggers = process_objects(data['loggers'], dic)
        if not isinstance(loggers, list):
            loggers = [loggers]
        samples = data['samples']
        return cls(data["id"], model, samples, loggers)
