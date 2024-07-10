torchtree.inference.sampler
===========================

.. py:module:: torchtree.inference.sampler


Classes
-------

.. autoapisummary::

   torchtree.inference.sampler.Sampler


Module Contents
---------------

.. py:class:: Sampler(id_: torchtree.typing.ID, model: torchtree.distributions.distributions.DistributionModel, samples: int, loggers: list[torchtree.core.logger.LoggerInterface])

   Bases: :py:obj:`torchtree.core.identifiable.Identifiable`, :py:obj:`torchtree.core.runnable.Runnable`


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

      :param dict[str, Any] data: dictionary representation of a Sampler object.
      :param dict[str, Identifiable] dic: dictionary containing torchtree objects
          keyed by their ID.
      :return: a :class:`~torchtree.inference.sampler.Sampler` object.



