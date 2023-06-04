:py:mod:`torchtree.evolution.tree_model_flexible`
=================================================

.. py:module:: torchtree.evolution.tree_model_flexible


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.evolution.tree_model_flexible.FlexibleTimeTreeModel




.. py:class:: FlexibleTimeTreeModel(id_: torchtree.typing.ID, tree, taxa: torchtree.evolution.taxa.Taxa, internal_heights: torchtree.core.abstractparameter.AbstractParameter)



   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:method:: json_factory(id_: str, newick: str, internal_heights: Union[dict, list, str], taxa: Union[dict, list, str], **kwargs)
      :staticmethod:

      Factory for creating tree models in JSON format.

      :param id_: ID of the tree model
      :param newick: tree in newick format
      :param taxa: dictionary of taxa with attributes or str reference


      :key internal_heights_id:  ID of internal_heights
      :key internal_heights: internal node heights. Can be a list of floats,
      a dictionary corresponding to a transformed parameter, or a str corresponding
      to a reference

      :return: tree model in JSON format compatible with from_json class method


   .. py:method:: from_json(data, dic)
      :classmethod:



