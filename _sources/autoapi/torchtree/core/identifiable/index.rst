:py:mod:`torchtree.core.identifiable`
=====================================

.. py:module:: torchtree.core.identifiable

.. autoapi-nested-parse::

   Interface for identifiable objects.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.core.identifiable.Identifiable




.. py:class:: Identifiable(id_: Optional[str])


   Bases: :py:obj:`torchtree.core.serializable.JSONSerializable`, :py:obj:`abc.ABC`

   Abstract class making an object identifiable.

   :param str or None id_: identifier of object

   .. py:property:: id
      :type: Optional[str]

      Return the identifier.


