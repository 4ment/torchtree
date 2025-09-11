torchtree.distributions.gar
===========================

.. py:module:: torchtree.distributions.gar

.. autoapi-nested-parse::

   Gamma autoregressive model.



Classes
-------

.. autoapisummary::

   torchtree.distributions.gar.GammaAutoregressiveModel


Module Contents
---------------

.. py:class:: GammaAutoregressiveModel(id_: torchtree.typing.ID, x: torchtree.core.abstractparameter.AbstractParameter, shape: torchtree.core.abstractparameter.AbstractParameter)

   Bases: :py:obj:`torchtree.core.model.CallableModel`


   Gamma autoregressive model.

   Computes the log probability of a gamma autoregressive (GAR) model with
   :math:`x_i | x_{i-1} \sim \text{Gamma}(\alpha, \alpha / x_{i-1})`.

   The mean of :math:`x_i | x_{i-1}` is :math:`x_{i-1}` and the variance
   is :math:`x_{i-1}^2/\alpha`.

   :param id_: ID of GAR object.
   :type id_: str or None
   :param AbstractParameter x: latent state parameter.
   :param AbstractParameter shape: shape parameter.


   .. py:attribute:: x


   .. py:attribute:: shape


   .. py:method:: from_json(data: dict[str, Any], dic: dict[str, torchtree.core.identifiable.Identifiable]) -> GammaAutoregressiveModel
      :classmethod:


      Creates a gamma autoregressive model object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a gamma autoregressive model object.
      :param dict[str, Identifiable] dic: dictionary containing torchtree objects
          keyed by their ID.

      **JSON attributes**:

       Mandatory:
        - id (str): unique string identifier.
        - x (dict or str): latent state parameter.

       Optional:
        - shape (dict or str): shape parameter. (Default: 1.0)

      :example:
      >>> x = {"id": "x", "type": "Parameter", "tensor": [1., 2., 3.]}
      >>> gar_dic = {"id": "gar", "x": x}
      >>> gar = GammaAutoregressiveModel.from_json(gar_dic, {})
      >>> isinstance(gar, GammaAutoregressiveModel)
      True



