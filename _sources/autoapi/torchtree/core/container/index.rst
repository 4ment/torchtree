:py:mod:`torchtree.core.container`
==================================

.. py:module:: torchtree.core.container


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.core.container.Container




.. py:class:: Container(id_: Optional[str], objects: list[Union[torchtree.core.model.Model, torchtree.core.abstractparameter.AbstractParameter]])



   Container for multiple objects of type Model or AbstractParameter.

   This class inherits from Model so an object referencing this object should be
   listening for model updates (class inherits from ModelListener).

   :param id_: ID of objects
   :param objects: list of Models or AbstractParameters

   .. py:method:: params()


   .. py:method:: callables()


   .. py:method:: handle_model_changed(model, obj, index) -> None


   .. py:method:: handle_parameter_changed(variable, index, event) -> None


   .. py:method:: from_json(data, dic)
      :classmethod:
      :abstractmethod:



