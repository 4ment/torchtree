:py:mod:`torchtree.evolution.alignment`
=======================================

.. py:module:: torchtree.evolution.alignment


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.evolution.alignment.Alignment



Functions
~~~~~~~~~

.. autoapisummary::

   torchtree.evolution.alignment.read_fasta_sequences
   torchtree.evolution.alignment.calculate_frequencies
   torchtree.evolution.alignment.calculate_frequencies_per_codon_position
   torchtree.evolution.alignment.calculate_F3x4_from_nucleotide
   torchtree.evolution.alignment.calculate_F3x4
   torchtree.evolution.alignment.calculate_substitutions
   torchtree.evolution.alignment.calculate_ts_tv
   torchtree.evolution.alignment.calculate_kappa



Attributes
~~~~~~~~~~

.. autoapisummary::

   torchtree.evolution.alignment.Sequence


.. py:data:: Sequence

   

.. py:class:: Alignment(id_: torchtree.typing.ID, sequences: list[Sequence], taxa: torchtree.evolution.taxa.Taxa, data_type: torchtree.evolution.datatype.DataType)

   Bases: :py:obj:`torchtree.core.model.Identifiable`, :py:obj:`collections.UserList`

   Sequence alignment.

   :param id_: ID of object
   :param sequences: list of sequences
   :param taxa: Taxa object

   .. py:property:: sequence_size
      :type: int


   .. py:property:: taxa
      :type: torchtree.evolution.taxa.Taxa


   .. py:property:: data_type
      :type: torchtree.evolution.datatype.DataType


   .. py:method:: get(id_: torchtree.typing.ID, filename: str, taxa: torchtree.evolution.taxa.Taxa) -> Alignment
      :classmethod:


   .. py:method:: from_json(data, dic)
      :classmethod:



.. py:function:: read_fasta_sequences(filename: str) -> list[Sequence]


.. py:function:: calculate_frequencies(alignment: Alignment)


.. py:function:: calculate_frequencies_per_codon_position(alignment: Alignment)


.. py:function:: calculate_F3x4_from_nucleotide(data_type, nuc_freq)


.. py:function:: calculate_F3x4(alignment)


.. py:function:: calculate_substitutions(alignment: Alignment, mapping)


.. py:function:: calculate_ts_tv(alignment: Alignment)


.. py:function:: calculate_kappa(alignment, freqs)


