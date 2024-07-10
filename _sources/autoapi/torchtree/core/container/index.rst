torchtree.core.container
========================

.. py:module:: torchtree.core.container


Classes
-------

.. autoapisummary::

   torchtree.core.container.Container


Module Contents
---------------

.. py:class:: Container(id_: Optional[str], objects: list[Union[torchtree.core.model.Model, torchtree.core.abstractparameter.AbstractParameter]])

   Bases: :py:obj:`torchtree.core.model.Model`


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


      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



