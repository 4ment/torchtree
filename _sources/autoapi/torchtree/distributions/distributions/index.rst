:py:mod:`torchtree.distributions.distributions`
===============================================

.. py:module:: torchtree.distributions.distributions


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.distributions.distributions.DistributionModel
   torchtree.distributions.distributions.Distribution




.. py:class:: DistributionModel(id_: Optional[str])



   Classes inheriting from :class:`Model` and
   :class:`collections.abc.Callable`.

   CallableModel are Callable and the returned value is cached in case
   we need to use this value multiple times without the need to
   recompute it.

   .. py:method:: rsample(sample_shape=torch.Size()) -> None
      :abstractmethod:


   .. py:method:: sample(sample_shape=torch.Size()) -> None
      :abstractmethod:


   .. py:method:: log_prob(x: torchtree.core.abstractparameter.AbstractParameter = None) -> torch.Tensor
      :abstractmethod:


   .. py:method:: entropy() -> torch.Tensor
      :abstractmethod:



.. py:class:: Distribution(id_: Optional[str], dist: Type[torch.distributions.Distribution], x: Union[list[torchtree.core.abstractparameter.AbstractParameter], torchtree.core.abstractparameter.AbstractParameter], args: OrderedDict[str, AbstractParameter], **kwargs)



   Wrapper for torch Distribution.

   :param id_: ID of joint distribution
   :param dist: class of torch Distribution
   :param x: random variable to evaluate/sample using distribution
   :param args: parameters of the distribution
   :param **kwargs: optional arguments for instanciating torch Distribution

   .. py:property:: event_shape
      :type: torch.Size


   .. py:property:: batch_shape
      :type: torch.Size


   .. py:method:: rsample(sample_shape=torch.Size()) -> None


   .. py:method:: sample(sample_shape=torch.Size()) -> None


   .. py:method:: log_prob(x: Union[list[torchtree.core.abstractparameter.AbstractParameter], torchtree.core.abstractparameter.AbstractParameter] = None) -> torch.Tensor


   .. py:method:: entropy() -> torch.Tensor


   .. py:method:: json_factory(id_: str, distribution: str, x: Union[str, dict], parameters: Union[str, dict] = None) -> dict
      :staticmethod:


   .. py:method:: from_json(data, dic)
      :classmethod:



