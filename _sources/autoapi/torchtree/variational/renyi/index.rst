:py:mod:`torchtree.variational.renyi`
=====================================

.. py:module:: torchtree.variational.renyi


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.variational.renyi.VR




.. py:class:: VR(id_: torchtree.typing.ID, q: torchtree.distributions.distributions.DistributionModel, p: torchtree.core.model.CallableModel, samples: torch.Size, alpha: float)



   Class representing the variational Renyi bound (VR) [#Li2016]_.
   VR extends traditional variational inference to Rényi’s :math:`\alpha`-divergences.

   :param id_: unique identifier of object.
   :type id_: str or None
   :param DistributionModel q: variational distribution.
   :param CallableModel p: joint distribution.
   :param torch.Size samples: number of samples to form estimator.
   :param float alpha: order of :math:`\alpha`-divergence.

   .. [#Li2016] Yingzhen Li, Richard E. Turner. Rényi Divergence Variational Inference.

   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: from_json(data, dic) -> VR
      :classmethod:



