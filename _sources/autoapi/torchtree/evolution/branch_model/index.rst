torchtree.evolution.branch_model
================================

.. py:module:: torchtree.evolution.branch_model


Classes
-------

.. autoapisummary::

   torchtree.evolution.branch_model.BranchModel
   torchtree.evolution.branch_model.AbstractClockModel
   torchtree.evolution.branch_model.StrictClockModel
   torchtree.evolution.branch_model.SimpleClockModel


Module Contents
---------------

.. py:class:: BranchModel(id_: Optional[str])

   Bases: :py:obj:`torchtree.core.model.Model`


   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.


   .. py:property:: rates
      :abstractmethod:



.. py:class:: AbstractClockModel(id_: torchtree.typing.ID, rates: torchtree.core.abstractparameter.AbstractParameter, tree: torchtree.evolution.tree_model.TreeModel)

   Bases: :py:obj:`BranchModel`, :py:obj:`abc.ABC`


   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.


   .. py:attribute:: tree


   .. py:method:: handle_model_changed(model, obj, index)


   .. py:method:: handle_parameter_changed(variable, index, event)


.. py:class:: StrictClockModel(id_: torchtree.typing.ID, rates: torchtree.core.abstractparameter.AbstractParameter, tree: torchtree.evolution.tree_model.TreeModel)

   Bases: :py:obj:`AbstractClockModel`


   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.


   .. py:attribute:: branch_count


   .. py:property:: rates
      :type: torch.Tensor



   .. py:method:: from_json(data, dic)
      :classmethod:


      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: SimpleClockModel(id_: torchtree.typing.ID, rates: torchtree.core.abstractparameter.AbstractParameter, tree: torchtree.evolution.tree_model.TreeModel)

   Bases: :py:obj:`AbstractClockModel`


   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.


   .. py:property:: rates
      :type: torch.Tensor



   .. py:method:: json_factory(id_: str, tree_model, rate)
      :staticmethod:



   .. py:method:: from_json(data, dic)
      :classmethod:


      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



