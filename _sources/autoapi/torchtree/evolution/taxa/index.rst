:py:mod:`torchtree.evolution.taxa`
==================================

.. py:module:: torchtree.evolution.taxa


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.evolution.taxa.Taxon
   torchtree.evolution.taxa.Taxa




.. py:class:: Taxon(id_, attributes)


   Bases: :py:obj:`torchtree.core.model.Identifiable`, :py:obj:`collections.UserDict`

   Abstract class making an object identifiable.

   :param str or None id_: identifier of object

   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: Taxa(id_, taxa)


   Bases: :py:obj:`torchtree.core.model.Identifiable`, :py:obj:`collections.UserList`

   Abstract class making an object identifiable.

   :param str or None id_: identifier of object

   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



