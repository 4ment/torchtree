:py:mod:`torchtree.inference.sampler`
=====================================

.. py:module:: torchtree.inference.sampler


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.inference.sampler.Sampler




.. py:class:: Sampler(model: torchtree.distributions.distributions.DistributionModel, samples: int, loggers: list[torchtree.core.logger.LoggerInterface])



   Class for sampling a distribution and optionally logging things.

   :param DistributionModel model: model to sample from.
   :param int samples: number of sample to draw.
   :param loggers: list of loggers.
   :type loggers: list[LoggerInterface]

   .. py:method:: run() -> None

      Run sampler: sample and log to loggers.


   .. py:method:: from_json(data, dic) -> Sampler
      :classmethod:

      Create a Sampler object.

      :param data: json representation of Sampler object.
      :type data: dict[str,Any]
      :param dic: dictionary containing additional objects that can be referenced
      in data.
      :type dic: dict[str,Any]

      :return: a :class:`~torchtree.inference.sampler.Sampler` object.
      :rtype: Sampler



