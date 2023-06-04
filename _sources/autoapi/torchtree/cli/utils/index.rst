:py:mod:`torchtree.cli.utils`
=============================

.. py:module:: torchtree.cli.utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   torchtree.cli.utils.convert_date_to_real
   torchtree.cli.utils.read_dates_from_csv
   torchtree.cli.utils.make_unconstrained



.. py:function:: convert_date_to_real(day, month, year)


.. py:function:: read_dates_from_csv(input_file, date_format=None)


.. py:function:: make_unconstrained(json_object: Union[dict, list]) -> tuple[list[str], list[dict]]

   Returns a list of constrained parameter IDs (str) with the corresponding
   parameters (dict)


