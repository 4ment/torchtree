:py:mod:`torchtree.evolution.substitution_model.amino_acid`
===========================================================

.. py:module:: torchtree.evolution.substitution_model.amino_acid


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.evolution.substitution_model.amino_acid.LG
   torchtree.evolution.substitution_model.amino_acid.WAG




.. py:class:: LG(id_: torchtree.typing.ID)


   Bases: :py:obj:`torchtree.evolution.substitution_model.general.EmpiricalSubstitutionModel`

   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: WAG(id_: torchtree.typing.ID)


   Bases: :py:obj:`torchtree.evolution.substitution_model.general.EmpiricalSubstitutionModel`

   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



