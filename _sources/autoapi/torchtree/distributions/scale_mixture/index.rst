:py:mod:`torchtree.distributions.scale_mixture`
===============================================

.. py:module:: torchtree.distributions.scale_mixture

.. autoapi-nested-parse::

   Scale mixture of Normal distributions.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.distributions.scale_mixture.ScaleMixtureNormal




.. py:class:: ScaleMixtureNormal(id_: torchtree.typing.ID, x: torchtree.core.abstractparameter.AbstractParameter, loc: Union[torchtree.core.abstractparameter.AbstractParameter, float], scale: torchtree.core.abstractparameter.AbstractParameter, gamma: torchtree.core.abstractparameter.AbstractParameter, slab: Union[torchtree.core.abstractparameter.AbstractParameter, float] = None)



   Scale mixture of Normal distributions.

   Regularized when a slab width parameter or scalar is provided [piironen2017]_

   :param id_: ID of object
   :param loc: mean of the distribution
   :param x: random variable
   :param scale: global scale
   :param gamma: local scale
   :param slab: slab width

   .. [piironen2017] Piironen and Vehtari 2017. Sparsity information and regularization
    in the horseshoe and other shrinkage priors.

   .. py:method:: handle_model_changed(model, obj, index) -> None


   .. py:method:: json_factory(id_: str, x, loc, global_scale, local_scale, slab=None)
      :staticmethod:


   .. py:method:: from_json(data: dict[str, Any], dic: dict[str, torchtree.core.identifiable.Identifiable]) -> ScaleMixtureNormal
      :classmethod:

      Creates a ScaleMixtureNormal object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a
          ScaleMixtureNormal object.
      :param dict[str, Identifiable] dic: dictionary containing torchtree objects
          keyed by their ID.

      **JSON attributes**:

       Mandatory:
        - id (str): unique string identifier.
        - x (dict or str): parameter.
        - loc (dict or str or float): location parameter.
        - scale (dict or str): global scale parameter.
        - local_scale (dict or str): local scale parameter.

       Optional:
        - slab (dict or str or float): slab parameter

      :example:
      >>> x = {"id": "x", "type": "Parameter", "tensor": [1., 2.]}
      >>> loc = {"id": "loc", "type": "Parameter", "tensor": [1.]}
      >>> global_scale = {"id": "global", "type": "Parameter", "tensor": [1.]}
      >>> local_scale = {"id": "local", "type": "Parameter", "tensor": [0.1, 0.2]}
      >>> mixture_dic = {"id": "mixture", "x": x, "loc": loc,
      ...     "global_scale": global_scale, "local_scale": local_scale}
      >>> mixture = ScaleMixtureNormal.from_json(mixture_dic, {})
      >>> isinstance(mixture, ScaleMixtureNormal)
      True
      >>> isinstance(mixture(), Tensor)
      True



