:py:mod:`torchtree.variational.chi`
===================================

.. py:module:: torchtree.variational.chi


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.variational.chi.CUBO




.. py:class:: CUBO(id_: torchtree.typing.ID, q: torchtree.distributions.distributions.DistributionModel, p: torchtree.core.model.CallableModel, samples: torch.Size, n: torch.Tensor)


   Bases: :py:obj:`torchtree.core.model.CallableModel`

   Class representing the :math:`\chi`-upper bound (CUBO) objective [#Dieng2017]_.

   :param id_: unique identifier of object.
   :type id_: str or None
   :param DistributionModel q: variational distribution.
   :param CallableModel p: joint distribution.
   :param torch.Size samples: number of samples to form estimator.
   :param torch.Tensor n: order of :math:`\chi`-divergence.

   .. [#Dieng2017] Adji Bousso Dieng, Dustin Tran, Rajesh Ranganath, John Paisley,
    David Blei. Variational Inference
    via :math:`\chi` Upper Bound Minimization

   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: from_json(data, dic) -> CUBO
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



