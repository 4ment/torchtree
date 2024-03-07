:py:mod:`torchtree.distributions.distributions`
===============================================

.. py:module:: torchtree.distributions.distributions

.. autoapi-nested-parse::

   torchtree distribution classes.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.distributions.distributions.DistributionModel
   torchtree.distributions.distributions.Distribution




.. py:class:: DistributionModel(id_: Optional[str])


   Bases: :py:obj:`torchtree.core.model.CallableModel`

   Abstract base class for distribution models.

   .. py:method:: rsample(sample_shape=torch.Size()) -> None
      :abstractmethod:

      Generates a sample_shape shaped reparameterized sample or
      sample_shape shaped batch of reparameterized samples if the
      distribution parameters are batched.


   .. py:method:: sample(sample_shape=torch.Size()) -> None
      :abstractmethod:

      Generates a sample_shape shaped sample or sample_shape shaped batch
      of samples if the distribution parameters are batched.


   .. py:method:: log_prob(x: torchtree.core.abstractparameter.AbstractParameter = None) -> torch.Tensor
      :abstractmethod:

      Returns the log of the probability density/mass function evaluated
      at x.

      :param Parameter x: value to evaluate
      :return: log probability
      :rtype: Tensor


   .. py:method:: entropy() -> torch.Tensor
      :abstractmethod:

      Returns entropy of distribution, batched over batch_shape.

      :return: Tensor of shape batch_shape.
      :rtype: Tensor



.. py:class:: Distribution(id_: Optional[str], dist: Type[torch.distributions.Distribution], x: Union[list[torchtree.core.abstractparameter.AbstractParameter], torchtree.core.abstractparameter.AbstractParameter], parameters: dict[str, torchtree.core.abstractparameter.AbstractParameter], **kwargs)


   Bases: :py:obj:`DistributionModel`

   Wrapper for :class:`torch.distributions.distribution.Distribution`.

   :param id_: ID of distribution
   :param dist: class of torch Distribution
   :param x: random variable to evaluate/sample using distribution
   :param dict[str, AbstractParameter] parameters: parameters of the distribution
   :param **kwargs: optional arguments for instanciating torch Distribution

   .. py:property:: event_shape
      :type: torch.Size


   .. py:property:: batch_shape
      :type: torch.Size


   .. py:property:: distribution
      :type: torch.distributions.Distribution


   .. py:method:: rsample(sample_shape=torch.Size()) -> None

      Generates a sample_shape shaped reparameterized sample or
      sample_shape shaped batch of reparameterized samples if the
      distribution parameters are batched.


   .. py:method:: sample(sample_shape=torch.Size()) -> None

      Generates a sample_shape shaped sample or sample_shape shaped batch
      of samples if the distribution parameters are batched.


   .. py:method:: log_prob(x: Union[list[torchtree.core.abstractparameter.AbstractParameter], torchtree.core.abstractparameter.AbstractParameter] = None) -> torch.Tensor

      Returns the log of the probability density/mass function evaluated
      at x.

      :param Parameter x: value to evaluate
      :return: log probability
      :rtype: Tensor


   .. py:method:: entropy() -> torch.Tensor

      Returns entropy of distribution, batched over batch_shape.

      :return: Tensor of shape batch_shape.
      :rtype: Tensor


   .. py:method:: json_factory(id_: str, distribution: str, x: Union[str, dict], parameters: Union[str, dict] = None) -> dict
      :staticmethod:


   .. py:method:: from_json(data: dict[str, Any], dic: dict[str, torchtree.core.identifiable.Identifiable]) -> Distribution
      :classmethod:

      Creates a Distribution object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a
          Distribution object.
      :param dict[str, Identifiable] dic: dictionary containing torchtree objects
          keyed by their ID.

      **JSON attributes**:

       Mandatory:
        - id (str): unique string identifier.
        - distribution (str): complete path to the torch distribution class,
          including package and module.
        - x (dict or str): parameter.

       Optional:
        - parameters (dict): parameters of the underlying torch Distribution.

      **JSON examples**:

      .. code-block:: json

        {
          "id": "exp",
          "distribution": "torch.distributions.Exponential",
          "x": {
              "id": "y",
              "type": "Parameter",
              "tensor": 0.1
          },
          "parameters": {
            "rate": {
              "id": "rate",
              "type": "Parameter",
              "tensor": 0.1
            }
          }
        }

      .. code-block:: json

        {
          "id": "normal",
          "distribution": "torch.distributions.Normal",
          "x": {
              "id": "y",
              "type": "Parameter",
              "tensor": 0.1
          },
          "parameters": {
            "loc": {
              "id": "loc",
              "type": "Parameter",
              "tensor": 0.0
            },
            "scale": {
              "id": "scale",
              "type": "Parameter",
              "tensor": 0.1
            }
          }
        }

      :example:
      >>> x_dict = {"id": "x", "type": "Parameter", "tensor": [1., 2.]}
      >>> x = Parameter.from_json(x_dict, {})
      >>> dic = {"x": x}
      >>> loc = {"id": "loc", "type": "Parameter", "tensor": [0.1]}
      >>> scale = {"id": "scale", "type": "Parameter", "tensor": [1.]}
      >>> normal_dic = {"id": "normal", "distribution": "torch.distributions.Normal",
      ...     "x": "x", "parameters":{"loc": loc, "scale": scale}}
      >>> normal = Distribution.from_json(normal_dic, dic)
      >>> isinstance(normal, Distribution)
      True
      >>> exp_dic = {"id": "exp", "x": "x", "parameters":{"rate": 1.0},
      ...     "distribution": "torch.distributions.Exponential"}
      >>> exp = Distribution.from_json(exp_dic, dic)
      >>> exp() == torch.distributions.Exponential(1.0).log_prob(x.tensor)
      tensor([True, True])

      .. note::
          The names of the keys in the `parameters` dictionary must match the
          variable names used in the signature of the torch distributions.
          See https://pytorch.org/docs/stable/distributions.html.



