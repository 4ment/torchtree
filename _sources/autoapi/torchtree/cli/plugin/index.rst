torchtree.cli.plugin
====================

.. py:module:: torchtree.cli.plugin


Classes
-------

.. autoapisummary::

   torchtree.cli.plugin.Plugin


Module Contents
---------------

.. py:class:: Plugin

   Bases: :py:obj:`abc.ABC`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: load_arguments(subparsers)
      :abstractmethod:



   .. py:method:: process_tree_likelihood(arg, data)


   .. py:method:: process_coalescent(arg, data)


   .. py:method:: process_all(arg, data)


