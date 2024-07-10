torchtree.core.serializable
===========================

.. py:module:: torchtree.core.serializable

.. autoapi-nested-parse::

   Interface for serializable objects.



Classes
-------

.. autoapisummary::

   torchtree.core.serializable.JSONSerializable


Module Contents
---------------

.. py:class:: JSONSerializable

   Bases: :py:obj:`abc.ABC`


   Interface making an object JSON serializable.

   Serializable base class establishing
   :meth:`~torch.core.serializable.JSONSerializable.from_json` abstract method.


   .. py:method:: from_json(data: dict[str, Any], dic: dict[str, Any]) -> Any
      :classmethod:

      :abstractmethod:


      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



   .. py:method:: from_json_safe(data: dict[str, Any], dic: dict[str, Any]) -> Any
      :classmethod:


      Parse dictionary to create object.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :raises JSONParseError: JSON error
      :return: torchtree object.
      :rtype: Any



