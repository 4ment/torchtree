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


   Bases: :py:obj:`torchtree.core.model.CallableModel`

   Class representing the variational Renyi bound.

   VR extends traditional variational inference to Rényi’s
   :math:`\alpha`-divergences :footcite:p:`li2016renyi`.

   :param str id_: identifier of object.
   :param DistributionModel q: variational distribution.
   :param CallableModel p: joint distribution.
   :param torch.Size samples: number of samples to form estimator.
   :param float alpha: order of :math:`\alpha`-divergence.

   .. footbibliography::

   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: from_json(data: dict[str, Any], dic: dict[str, torchtree.core.identifiable.Identifiable]) -> VR
      :classmethod:

      Creates a VR object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a VR object.
      :param dict[str, Identifiable] dic: dictionary containing torchtree objects
          keyed by their ID.

      **JSON attributes**:

      Mandatory:
        - id (str): unique string identifier.
        - variational (dict or str): variational distribution.
        - joint (dict or str): joint distribution.

      Optional:
        - samples (int or list of ints): number of samples
        - alpha (float): order of :math:`\alpha`-divergence (Default: 0).



