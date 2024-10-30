torchtree.evolution.io
======================

.. py:module:: torchtree.evolution.io


Classes
-------

.. autoapisummary::

   torchtree.evolution.io.Node


Functions
---------

.. autoapisummary::

   torchtree.evolution.io.read_tree
   torchtree.evolution.io.read_tree_and_alignment
   torchtree.evolution.io.to_nexus
   torchtree.evolution.io.convert_samples_to_nexus
   torchtree.evolution.io.random_tree_from_heights
   torchtree.evolution.io.parse_translate
   torchtree.evolution.io.parse_trees
   torchtree.evolution.io.parse_taxa
   torchtree.evolution.io.split_newick
   torchtree.evolution.io.extract_taxa


Module Contents
---------------

.. py:function:: read_tree(tree, dated=True, heterochornous=True)

.. py:function:: read_tree_and_alignment(tree, alignment, dated=True, heterochornous=True)

.. py:function:: to_nexus(node, fp)

.. py:function:: convert_samples_to_nexus(tree, sample, output)

.. py:class:: Node(name, height=0.0)

   .. py:attribute:: name


   .. py:attribute:: height


   .. py:attribute:: parent
      :value: None



   .. py:attribute:: children
      :value: []



.. py:function:: random_tree_from_heights(sampling: torch.Tensor, heights: torch.Tensor) -> Node

.. py:function:: parse_translate(fp)

.. py:function:: parse_trees(fp, count=None)

.. py:function:: parse_taxa(fp)

.. py:function:: split_newick(newick: str) -> list[str]

   Split tree in newick format around (),;

   .. rubric:: Example

   >>> newick = '((a:1[&a={1,2}],b:1):1,c:1);'
   >>> split_newick('((a:1,b:1):1,c:1);')
   ['(', '(', 'a:1', ',', 'b:1', ')', ':1', ',', 'c:1', ')', ';']

   :param str newick: newick tree
   :return List[str]: list of strings


.. py:function:: extract_taxa(file_name: str) -> list[str]

   Extract taxon list from a nexus file.

   This function will try get the taxon names from the taxa and trees blocks.

   :param str file_name: path to the nexus file
   :return List[str]: list of taxon names


