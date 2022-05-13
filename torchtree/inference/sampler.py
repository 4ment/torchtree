from __future__ import annotations

from ..core.logger import LoggerInterface
from ..core.runnable import Runnable
from ..core.serializable import JSONSerializable
from ..core.utils import process_object, process_objects, register_class
from ..distributions.distributions import DistributionModel


@register_class
class Sampler(JSONSerializable, Runnable):
    r"""
    Class for sampling a distribution and optionally logging things.

    :param DistributionModel model: model to sample from.
    :param int samples: number of sample to draw.
    :param loggers: list of loggers.
    :type loggers: list[LoggerInterface]
    """

    def __init__(
        self, model: DistributionModel, samples: int, loggers: list[LoggerInterface]
    ) -> None:
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

        :param data: json representation of Sampler object.
        :type data: dict[str,Any]
        :param dic: dictionary containing additional objects that can be referenced
        in data.
        :type dic: dict[str,Any]

        :return: a :class:`~torchtree.inference.sampler.Sampler` object.
        :rtype: Sampler
        """
        model = process_object(data['model'], dic)
        loggers = process_objects(data['loggers'], dic)
        if not isinstance(loggers, list):
            loggers = [loggers]
        samples = data['samples']
        return cls(model, samples, loggers)
