:py:mod:`torchtree.nf.energy_functions`
=======================================

.. py:module:: torchtree.nf.energy_functions


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.nf.energy_functions.EnergyFunctionModel



Functions
~~~~~~~~~

.. autoapisummary::

   torchtree.nf.energy_functions.w1
   torchtree.nf.energy_functions.w2
   torchtree.nf.energy_functions.w3



.. py:function:: w1(z)


.. py:function:: w2(z)


.. py:function:: w3(z)


.. py:class:: EnergyFunctionModel(id_: torchtree.typing.ID, x: torchtree.core.abstractparameter.AbstractParameter, desc: str, dtype=None, device=None)


   Bases: :py:obj:`torchtree.core.model.CallableModel`

   Classes inheriting from :class:`Model` and
   :class:`collections.abc.Callable`.

   CallableModel are Callable and the returned value is cached in case
   we need to use this value multiple times without the need to
   recompute it.

   .. py:method:: handle_model_changed(model, obj, index)


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



