:py:mod:`torchtree.distributions.bayesian_bridge`
=================================================

.. py:module:: torchtree.distributions.bayesian_bridge

.. autoapi-nested-parse::

   Bayesian bridge prior.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.distributions.bayesian_bridge.BayesianBridge



Functions
~~~~~~~~~

.. autoapisummary::

   torchtree.distributions.bayesian_bridge.process_object_number



.. py:class:: BayesianBridge(id_: torchtree.typing.ID, x: torchtree.core.abstractparameter.AbstractParameter, scale: Union[torchtree.core.abstractparameter.AbstractParameter, torch.Tensor], alpha: Union[torchtree.core.abstractparameter.AbstractParameter, torch.Tensor] = None, local_scale: Union[torchtree.core.abstractparameter.AbstractParameter, torch.Tensor] = None, slab: Union[torchtree.core.abstractparameter.AbstractParameter, torch.Tensor] = None)



   Bayesian bridge.

   [polson2014]_ and [nishimura2019]_

   :param id_: ID of object
   :param x: random variable
   :param scale: global scale
   :param alpha: exponent
   :param local_scale: local scale
   :param slab: slab width

   .. [polson2014] Polson and Scott 2014. The Bayesian Bridge.
   .. [nishimura2019] Nishimura, Suchard 2019 .Shrinkage with shrunken shoulders: Gibbs
       sampling shrinkage model posteriors with guaranteed convergence rates.

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


