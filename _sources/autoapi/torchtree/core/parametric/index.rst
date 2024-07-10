torchtree.core.parametric
=========================

.. py:module:: torchtree.core.parametric


Classes
-------

.. autoapisummary::

   torchtree.core.parametric.ModelListener
   torchtree.core.parametric.ParameterListener
   torchtree.core.parametric.Parametric


Module Contents
---------------

.. py:class:: ModelListener

   Bases: :py:obj:`abc.ABC`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: handle_model_changed(model, obj, index) -> None
      :abstractmethod:



.. py:class:: ParameterListener

   Bases: :py:obj:`abc.ABC`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: handle_parameter_changed(variable: torchtree.core.abstractparameter.AbstractParameter, index, event) -> None
      :abstractmethod:



.. py:class:: Parametric

   Bases: :py:obj:`ModelListener`, :py:obj:`ParameterListener`, :py:obj:`abc.ABC`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: register_parameter(name: str, parameter: torchtree.core.abstractparameter.AbstractParameter) -> None


   .. py:method:: register_model(name: str, model: Parametric) -> None


   .. py:method:: parameters() -> list[torchtree.core.abstractparameter.AbstractParameter]

      Returns parameters of instance Parameter.



