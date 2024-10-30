torchtree.core.logger
=====================

.. py:module:: torchtree.core.logger


Classes
-------

.. autoapisummary::

   torchtree.core.logger.LoggerInterface
   torchtree.core.logger.Logger
   torchtree.core.logger.TreeLogger
   torchtree.core.logger.CSV
   torchtree.core.logger.Dumper
   torchtree.core.logger.ContainerLogger


Module Contents
---------------

.. py:class:: LoggerInterface

   Bases: :py:obj:`torchtree.core.serializable.JSONSerializable`, :py:obj:`torchtree.core.runnable.Runnable`


   Interface for logging things like parameters or trees to a file.


   .. py:method:: log(*args, **kwargs) -> None
      :abstractmethod:



   .. py:method:: initialize() -> None
      :abstractmethod:



   .. py:method:: close() -> None
      :abstractmethod:



   .. py:method:: run() -> None


.. py:class:: Logger(objs: list[Union[torchtree.core.abstractparameter.AbstractParameter, torchtree.core.model.CallableModel]], every: int, **kwargs)

   Bases: :py:obj:`LoggerInterface`


   Class for logging Parameter objects to a file.

   :param objs: list of Parameter or CallableModel objects
   :type objs: list[Parameter or CallableModel]
   :param int every: logging frequency
   :param kwargs: optionals


   .. py:attribute:: every


   .. py:attribute:: kwargs


   .. py:attribute:: objs


   .. py:attribute:: f
      :value: None



   .. py:attribute:: writer
      :value: None



   .. py:attribute:: sample
      :value: 1



   .. py:method:: initialize() -> None


   .. py:method:: log(*args, **kwargs) -> None


   .. py:method:: close() -> None


   .. py:method:: from_json(data, dic) -> Logger
      :classmethod:


      Create a Logger object.

      :param data: json representation of Logger object.
      :type data: dict[str,Any]
      :param dic: dictionary containing additional objects that can be
              referenced in data.
      :type dic: dict[str,Any]
      :return: a
      :class: `~torchtree.core.logger.Logger` object.
      :rtype: Logger



.. py:class:: TreeLogger(tree_model: torchtree.evolution.tree_model.TreeModel, every: int, **kwargs)

   Bases: :py:obj:`LoggerInterface`


   Class for logging trees to a file.

   :param TreeModel objs: TreeModel object
   :param int every: logging frequency
   :param kwargs: optionals


   .. py:attribute:: tree_model


   .. py:attribute:: every


   .. py:attribute:: file_name


   .. py:attribute:: kwargs


   .. py:attribute:: sample
      :value: 1



   .. py:attribute:: f
      :value: None



   .. py:method:: initialize() -> None


   .. py:method:: log(*args, **kwargs) -> None


   .. py:method:: close() -> None


   .. py:method:: from_json(data, dic) -> TreeLogger
      :classmethod:


      Create a TreeLogger object.

      :param data: json representation of TreeLogger object.
      :type data: dict[str,Any]
      :param dic: dictionary containing additional objects that can be
              referenced in data.
      :type dic: dict[str,Any]
      :return: a
      :class: `~torchtree.core.logger.TreeLogger` object.
      :rtype: TreeLogger



.. py:class:: CSV(objs: list[torchtree.core.abstractparameter.AbstractParameter], **kwargs)

   Bases: :py:obj:`torchtree.core.serializable.JSONSerializable`, :py:obj:`torchtree.core.runnable.Runnable`


   Class for writting parameters to a CSV file.

   :param objs: list of Parameter objects
   :type objs: list[Parameter]


   .. py:attribute:: objs


   .. py:attribute:: file_name


   .. py:attribute:: kwargs


   .. py:method:: run() -> None


   .. py:method:: from_json(data, dic) -> CSV
      :classmethod:


      Create a CSV object.

      :param data: json representation of CSV object.
      :type data: dict[str,Any]
      :param dic: dictionary containing additional objects that can be
              referenced in data.
      :type dic: dict[str,Any]
      :return: a
      :class: `~torchtree.core.logger.CSV` object.
      :rtype: CSV



.. py:class:: Dumper(parameters: list[torchtree.core.abstractparameter.AbstractParameter], **kwargs)

   Bases: :py:obj:`torchtree.core.serializable.JSONSerializable`, :py:obj:`torchtree.core.runnable.Runnable`


   Class for saving parameters to a json file.

   :param parameters: list of Parameters.
   :type parameters: list[Parameter]


   .. py:attribute:: kwargs


   .. py:attribute:: parameters


   .. py:method:: run() -> None

      Write the parameters to the file.



   .. py:method:: from_json(data, dic) -> Dumper
      :classmethod:


      Create a Dumper object.

      :param data: json representation of Dumper object.
      :type data: dict[str,Any]
      :param dic: dictionary containing additional objects that can be
              referenced in data.
      :type dic: dict[str,Any]
      :return: a
      :class: `~torchtree.core.logger.Dumper` object.
      :rtype: Dumper



.. py:class:: ContainerLogger(inputs: list[Union[torchtree.core.abstractparameter.AbstractParameter, torchtree.core.model.CallableModel]], container, every: int)

   Bases: :py:obj:`LoggerInterface`


   Class for logging Parameter and CallableModel values to a list.

   :param inputs: list of Parameter or CallableModel objects
   :type inputs: list[Parameter or CallableModel]
   :param int every: logging frequency


   .. py:attribute:: container


   .. py:attribute:: every


   .. py:attribute:: inputs


   .. py:attribute:: sample
      :value: 1



   .. py:method:: initialize() -> None


   .. py:method:: log(*args, **kwargs) -> None


   .. py:method:: close() -> None


   .. py:method:: from_json(data, dic) -> ContainerLogger
      :classmethod:


      Create a ContainerLogger object.

      :param dict[str, Any] data: dictionary representation of a ContainerLogger
          object.
      :param dict[str, Identifiable] dic: dictionary containing torchtree objects
          keyed by their ID.

      **JSON attributes**:

       Mandatory:
        - container (list): python list to log values.
        - inputs (list[AbstractParameter or CallableObject]): list of parameters
          or models to log.

       Optional:
        - every (int): logging frequency.



