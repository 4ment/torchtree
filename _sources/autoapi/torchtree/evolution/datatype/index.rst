torchtree.evolution.datatype
============================

.. py:module:: torchtree.evolution.datatype


Classes
-------

.. autoapisummary::

   torchtree.evolution.datatype.DataType
   torchtree.evolution.datatype.AbstractDataType
   torchtree.evolution.datatype.NucleotideDataType
   torchtree.evolution.datatype.AminoAcidDataType
   torchtree.evolution.datatype.CodonDataType
   torchtree.evolution.datatype.GeneralDataType


Module Contents
---------------

.. py:class:: DataType(id_: Optional[str])

   Bases: :py:obj:`torchtree.core.model.Identifiable`, :py:obj:`abc.ABC`


   Abstract class making an object identifiable.

   :param str or None id_: identifier of object


   .. py:property:: states
      :type: tuple[str, Ellipsis]

      :abstractmethod:



   .. py:property:: state_count
      :type: int

      :abstractmethod:



   .. py:method:: encoding(string: str) -> int
      :abstractmethod:



   .. py:method:: partial(string: str, use_ambiguities=True) -> tuple[float, Ellipsis]
      :abstractmethod:



   .. py:property:: size
      :type: int

      :abstractmethod:



.. py:class:: AbstractDataType(id_: torchtree.typing.ID, states: tuple[str, Ellipsis])

   Bases: :py:obj:`DataType`, :py:obj:`abc.ABC`


   Abstract class making an object identifiable.

   :param str or None id_: identifier of object


   .. py:property:: states
      :type: tuple[str, Ellipsis]



   .. py:property:: state_count
      :type: int



   .. py:property:: size
      :type: int



.. py:class:: NucleotideDataType(id_: torchtree.typing.ID)

   Bases: :py:obj:`AbstractDataType`


   Abstract class making an object identifiable.

   :param str or None id_: identifier of object


   .. py:attribute:: NUCLEOTIDES
      :value: 'ACGTUKMRSWYBDHVN?-'



   .. py:attribute:: NUCLEOTIDE_STATES
      :value: (17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,...



   .. py:attribute:: NUCLEOTIDE_AMBIGUITY_STATES
      :value: ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0), (0.0,...



   .. py:method:: encoding(string) -> int


   .. py:method:: partial(string: str, use_ambiguities=True) -> tuple[float, Ellipsis]


   .. py:method:: from_json(data, dic)
      :classmethod:


      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: AminoAcidDataType(id_: torchtree.typing.ID)

   Bases: :py:obj:`AbstractDataType`


   Abstract class making an object identifiable.

   :param str or None id_: identifier of object


   .. py:attribute:: AMINO_ACIDS
      :value: 'ACDEFGHIKLMNPQRSTVWYBZX*?-'



   .. py:attribute:: AMINO_ACIDS_STATES
      :value: (25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25,...



   .. py:attribute:: AMINO_ACIDS_AMBIGUITY_STATES


   .. py:attribute:: AMINO_ACIDS_AMBIGUITY_STATES


   .. py:method:: encoding(string) -> int


   .. py:method:: partial(string: str, use_ambiguities=True) -> tuple[float, Ellipsis]


   .. py:method:: from_json(data, dic)
      :classmethod:


      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: CodonDataType(id_: torchtree.typing.ID, genetic_code: str)

   Bases: :py:obj:`AbstractDataType`


   Abstract class making an object identifiable.

   :param str or None id_: identifier of object


   .. py:attribute:: GENETIC_CODE_TABLES
      :value: ('KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF',...



   .. py:attribute:: GENETIC_CODE_NAMES
      :value: ('Universal', 'Vertebrate Mitochondrial', 'Yeast', 'Mold Protozoan Mitochondrial', 'Mycoplasma',...



   .. py:attribute:: NUMBER_OF_CODONS
      :value: (61, 60, 62, 62, 62, 62, 63, 62, 62, 61, 61, 62, 63, 62, 64)



   .. py:attribute:: CODON_TRIPLETS
      :value: ('AAA', 'AAC', 'AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 'AGA', 'AGC', 'AGG', 'AGT', 'ATA',...



   .. py:method:: encoding(codon) -> int


   .. py:method:: partial(string: str, use_ambiguities=True) -> tuple[float, Ellipsis]


   .. py:method:: from_json(data, dic)
      :classmethod:


      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: GeneralDataType(id_: torchtree.typing.ID, codes: tuple[str, Ellipsis], ambiguities: dict = {})

   Bases: :py:obj:`AbstractDataType`


   Abstract class making an object identifiable.

   :param str or None id_: identifier of object


   .. py:method:: encoding(string: str) -> int


   .. py:method:: partial(string: str, use_ambiguities=True) -> tuple[float, Ellipsis]


   .. py:method:: from_json(data, dic)
      :classmethod:


      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



