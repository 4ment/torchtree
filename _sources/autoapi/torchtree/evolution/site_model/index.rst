:py:mod:`torchtree.evolution.site_model`
========================================

.. py:module:: torchtree.evolution.site_model


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.evolution.site_model.SiteModel
   torchtree.evolution.site_model.ConstantSiteModel
   torchtree.evolution.site_model.InvariantSiteModel
   torchtree.evolution.site_model.UnivariateDiscretizedSiteModel
   torchtree.evolution.site_model.WeibullSiteModel
   torchtree.evolution.site_model.LogNormalSiteModel




.. py:class:: SiteModel(id_: torchtree.typing.ID, mu: torchtree.core.abstractparameter.AbstractParameter = None)



   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:method:: rates() -> torch.Tensor
      :abstractmethod:


   .. py:method:: probabilities() -> torch.Tensor
      :abstractmethod:



.. py:class:: ConstantSiteModel(id_: torchtree.typing.ID, mu: torchtree.core.abstractparameter.AbstractParameter = None)



   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:method:: rates() -> torch.Tensor


   .. py:method:: probabilities() -> torch.Tensor


   .. py:method:: handle_model_changed(model, obj, index)


   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None) -> None


   .. py:method:: cpu() -> None


   .. py:method:: from_json(data, dic)
      :classmethod:



.. py:class:: InvariantSiteModel(id_: torchtree.typing.ID, invariant: torchtree.core.abstractparameter.AbstractParameter, mu: torchtree.core.abstractparameter.AbstractParameter = None)



   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:property:: invariant
      :type: torch.Tensor


   .. py:method:: update_rates_probs(invariant: torch.Tensor)


   .. py:method:: rates() -> torch.Tensor


   .. py:method:: probabilities() -> torch.Tensor


   .. py:method:: handle_model_changed(model, obj, index)


   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: from_json(data, dic)
      :classmethod:



.. py:class:: UnivariateDiscretizedSiteModel(id_: torchtree.typing.ID, parameter: torchtree.core.abstractparameter.AbstractParameter, categories: int, invariant: torchtree.core.abstractparameter.AbstractParameter = None, mu: torchtree.core.abstractparameter.AbstractParameter = None)



   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:property:: invariant
      :type: torch.Tensor


   .. py:method:: inverse_cdf(parameter: torch.Tensor, quantile: torch.Tensor, invariant: torch.Tensor) -> torch.Tensor
      :abstractmethod:


   .. py:method:: update_rates(parameter: torch.Tensor, invariant: torch.Tensor)


   .. py:method:: rates() -> torch.Tensor


   .. py:method:: probabilities() -> torch.Tensor


   .. py:method:: handle_model_changed(model, obj, index)


   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None)


   .. py:method:: cpu() -> None



.. py:class:: WeibullSiteModel(id_: torchtree.typing.ID, parameter: torchtree.core.abstractparameter.AbstractParameter, categories: int, invariant: torchtree.core.abstractparameter.AbstractParameter = None, mu: torchtree.core.abstractparameter.AbstractParameter = None)



   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:property:: shape
      :type: torch.Tensor


   .. py:method:: inverse_cdf(parameter, quantile, invariant)


   .. py:method:: from_json(data, dic)
      :classmethod:



.. py:class:: LogNormalSiteModel(id_: torchtree.typing.ID, parameter: torchtree.core.abstractparameter.AbstractParameter, categories: int, invariant: torchtree.core.abstractparameter.AbstractParameter = None, mu: torchtree.core.abstractparameter.AbstractParameter = None)



   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:property:: scale
      :type: torch.Tensor


   .. py:method:: update_rates(value)


   .. py:method:: from_json(data, dic)
      :classmethod:



