torchtree.distributions.bayesian_bridge
=======================================

.. py:module:: torchtree.distributions.bayesian_bridge

.. autoapi-nested-parse::

   Bayesian bridge prior.



Classes
-------

.. autoapisummary::

   torchtree.distributions.bayesian_bridge.BayesianBridge


Functions
---------

.. autoapisummary::

   torchtree.distributions.bayesian_bridge.process_object_number


Module Contents
---------------

.. py:class:: BayesianBridge(id_: torchtree.typing.ID, x: torchtree.core.abstractparameter.AbstractParameter, scale: Union[torchtree.core.abstractparameter.AbstractParameter, torch.Tensor], alpha: Union[torchtree.core.abstractparameter.AbstractParameter, torch.Tensor] = None, local_scale: Union[torchtree.core.abstractparameter.AbstractParameter, torch.Tensor] = None, slab: Union[torchtree.core.abstractparameter.AbstractParameter, torch.Tensor] = None)

   Bases: :py:obj:`torchtree.core.model.CallableModel`


   Bayesian bridge prior.

   Creates a Bayesian bridge prior :footcite:p:`polson2014bayesian`.
   This class also implements the regularized version of the
   prior :footcite:p:`nishimura2023shrinkage`.

   :param str or None id_: ID of BayesianBridge object.
   :param AbstractParameter x: random variable.
   :param AbstractParameter or Tensor scale: global scale.
   :param AbstractParameter or Tensor alpha: exponent.
   :param AbstractParameter or Tensor local_scale: local scale.
   :param AbstractParameter or Tensor slab: slab width.

   .. footbibliography::


   .. py:method:: handle_model_changed(model, obj, index) -> None


   .. py:method:: json_factory(id_: str, x, scale, alpha)
      :staticmethod:



   .. py:method:: from_json(data: dict[str, Any], dic: dict[str, torchtree.core.identifiable.Identifiable]) -> BayesianBridge
      :classmethod:


      Creates a BayesianBridge object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a
          BayesianBridge object.
      :param dict[str, Identifiable] dic: dictionary containing torchtree objects
          keyed by their ID.

      **JSON attributes**:

       Mandatory:
        - id (str): unique string identifier.
        - x (dict or str): parameter.
        - scale (dict or str or float): global scale parameter.

       Optional:
        - alpha (dict or str or float): alpha parameter.
        - local_scale (dict or str or float): local scale parameter.
        - slab (dict or str or float): slab parameter

      :example:
      >>> x = {"id": "x", "type": "Parameter", "tensor": [1., 2., 3.]}
      >>> scale = {"id": "scale", "type": "Parameter", "tensor": [1.]}
      >>> alpha = {"id": "alpha", "type": "Parameter", "tensor": [0.1]}
      >>> bridge_dic = {"id": "bridge", "x": x, "scale": scale, "alpha": alpha}
      >>> bridge = BayesianBridge.from_json(bridge_dic, {})
      >>> isinstance(bridge, BayesianBridge)
      True
      >>> isinstance(bridge(), torch.Tensor)
      True

      .. note::
          local_scale or alpha are optional parameters but only one of them can
          be specified at a time. The slab parameter must be specified if a
          local_scale parameter is specified.



.. py:function:: process_object_number(data, dic, **options) -> Union[torch.Tensor, torchtree.core.abstractparameter.AbstractParameter]

   Data can be a Number, str, or dict.


