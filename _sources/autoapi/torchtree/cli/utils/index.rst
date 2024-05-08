:py:mod:`torchtree.cli.utils`
=============================

.. py:module:: torchtree.cli.utils


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.cli.utils.CONSTRAINT



Functions
~~~~~~~~~

.. autoapisummary::

   torchtree.cli.utils.convert_date_to_real
   torchtree.cli.utils.read_dates_from_csv
   torchtree.cli.utils.make_unconstrained
   torchtree.cli.utils.length_of_tensor_in_dict_parameter
   torchtree.cli.utils.remove_constraints



Attributes
~~~~~~~~~~

.. autoapisummary::

   torchtree.cli.utils.CONSTRAINT_PREFIX


.. py:data:: CONSTRAINT_PREFIX
   :value: '@'

   

.. py:class:: CONSTRAINT


   Bases: :py:obj:`enum.Enum`

   Generic enumeration.

   Derive from this class to define new enumerations.

   .. py:attribute:: LOWER

      

   .. py:attribute:: UPPER

      

   .. py:attribute:: SIMPLEX

      


.. py:function:: convert_date_to_real(day, month, year)


.. py:function:: read_dates_from_csv(input_file, date_format=None)


.. py:function:: make_unconstrained(json_object: Union[dict, list]) -> tuple[list[str], list[dict]]

   Returns a list of constrained parameter IDs (str) with the corresponding
   parameters (dict)


.. py:function:: length_of_tensor_in_dict_parameter(param: dict) -> int


.. py:function:: remove_constraints(obj)

   Remove constraint keys starting with CONSTRAINST_PREFIX.


