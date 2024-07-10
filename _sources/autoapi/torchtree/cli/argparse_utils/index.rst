torchtree.cli.argparse_utils
============================

.. py:module:: torchtree.cli.argparse_utils


Functions
---------

.. autoapisummary::

   torchtree.cli.argparse_utils.zero_or_path
   torchtree.cli.argparse_utils.str_or_float
   torchtree.cli.argparse_utils.list_of_float
   torchtree.cli.argparse_utils.list_or_int


Module Contents
---------------

.. py:function:: zero_or_path(arg)

.. py:function:: str_or_float(arg, choices)

   Used by argparse when the argument can be either a number or a string
   from a prespecified list of options.


.. py:function:: list_of_float(arg, length)

   Used by argparse when the argument should be a list of floats.


.. py:function:: list_or_int(arg, min=1, max=math.inf)

   Used by argparse when the argument should be a list of ints or a int.


