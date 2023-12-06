:py:mod:`torchtree.evolution.tree_model`
========================================

.. py:module:: torchtree.evolution.tree_model


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.evolution.tree_model.TreeModel
   torchtree.evolution.tree_model.AbstractTreeModel
   torchtree.evolution.tree_model.UnRootedTreeModel
   torchtree.evolution.tree_model.TimeTreeModel
   torchtree.evolution.tree_model.ReparameterizedTimeTreeModel



Functions
~~~~~~~~~

.. autoapisummary::

   torchtree.evolution.tree_model.heights_to_branch_lengths
   torchtree.evolution.tree_model.setup_indexes
   torchtree.evolution.tree_model.setup_dates
   torchtree.evolution.tree_model.initialize_dates_from_taxa
   torchtree.evolution.tree_model.heights_from_branch_lengths
   torchtree.evolution.tree_model.parse_tree



.. py:function:: heights_to_branch_lengths(node_heights, bounds, indexing)


.. py:function:: setup_indexes(tree, indices_postorder=False)


.. py:function:: setup_dates(tree, heterochronous=False)


.. py:function:: initialize_dates_from_taxa(tree, taxa, tag='date')


.. py:function:: heights_from_branch_lengths(tree, eps=1e-06)


.. py:function:: parse_tree(taxa, data)


.. py:class:: TreeModel(id_: Optional[str])


   Bases: :py:obj:`torchtree.core.model.Model`

   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:property:: postorder
      :type: list[list[int]]
      :abstractmethod:


   .. py:property:: taxa
      :type: list[str]
      :abstractmethod:


   .. py:method:: branch_lengths() -> torch.Tensor
      :abstractmethod:


   .. py:method:: write_newick(steam, **kwargs) -> None
      :abstractmethod:



.. py:class:: AbstractTreeModel(id_: torchtree.typing.ID, tree, taxa: torchtree.evolution.taxa.Taxa)


   Bases: :py:obj:`TreeModel`, :py:obj:`abc.ABC`

   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:property:: postorder


   .. py:property:: taxa


   .. py:method:: update_traversals() -> None


   .. py:method:: handle_model_changed(model, obj, index)


   .. py:method:: as_newick(**kwargs)


   .. py:method:: write_newick(stream, **kwargs) -> None



.. py:class:: UnRootedTreeModel(id_: torchtree.typing.ID, tree, taxa: torchtree.evolution.taxa.Taxa, branch_lengths: torchtree.core.abstractparameter.AbstractParameter)


   Bases: :py:obj:`AbstractTreeModel`

   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:method:: branch_lengths() -> torch.Tensor


   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: json_factory(id_: str, newick: str, branch_lengths: Union[dict, list, str], taxa: Union[dict, list, str], **kwargs)
      :staticmethod:

      Factory for creating tree models in JSON format.

      :param id_: ID of the tree model
      :param newick: tree in newick format
      :param branch_lengths: branch lengths
      :param taxa: list dictionary of taxa with attributes or str reference


      :key branch_lengths_id:  ID of branch_lengths (default: branch_lengths)
      :key taxa_id:  ID of taxa (default: taxa)
      :key keep_branch_lengths: if True use branch lengths in newick tree

      :return: tree model in JSON format compatible with from_json class method


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: TimeTreeModel(id_: torchtree.typing.ID, tree, taxa: torchtree.evolution.taxa.Taxa, internal_heights: torchtree.core.abstractparameter.AbstractParameter)


   Bases: :py:obj:`AbstractTreeModel`

   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:property:: node_heights
      :type: torch.Tensor


   .. py:method:: update_leaf_heights() -> None


   .. py:method:: update_traversals()


   .. py:method:: branch_lengths() -> torch.Tensor

      Return branch lengths calculated from node heights.

      Branch lengths are indexed by node index on the distal side of
      the tree. For example branch_lengths[0] corresponds to the branch
      starting from taxon with index 0.

      :return: branch lengths of tree
      :rtype: torch.Tensor


   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None) -> None

      Move tensors to CUDA using torch.cuda.


   .. py:method:: cpu() -> None

      Move tensors to CPU memory using ~torch.cpu.


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

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: ReparameterizedTimeTreeModel(id_: torchtree.typing.ID, tree, taxa: torchtree.evolution.taxa.Taxa, ratios_root_height: torchtree.core.abstractparameter.AbstractParameter = None, shifts: torchtree.core.abstractparameter.AbstractParameter = None)


   Bases: :py:obj:`TimeTreeModel`, :py:obj:`torchtree.core.model.CallableModel`

   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:property:: node_heights
      :type: torch.Tensor


   .. py:method:: update_node_heights() -> None


   .. py:method:: handle_model_changed(model, obj, index) -> None


   .. py:method:: handle_parameter_changed(variable: torchtree.core.abstractparameter.AbstractParameter, index, event) -> None


   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None) -> None

      Move tensors to CUDA using torch.cuda.


   .. py:method:: cpu() -> None

      Move tensors to CPU memory using ~torch.cpu.


   .. py:method:: json_factory(id_: str, newick: str, taxa: Union[dict, list, str], ratios: Union[dict, list, str] = None, root_height: Union[dict, list, str] = None, shifts: Union[dict, list, str] = None, **kwargs)
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

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



