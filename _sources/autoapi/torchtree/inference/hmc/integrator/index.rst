:py:mod:`torchtree.inference.hmc.integrator`
============================================

.. py:module:: torchtree.inference.hmc.integrator


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.inference.hmc.integrator.Integrator
   torchtree.inference.hmc.integrator.LeapfrogIntegrator



Functions
~~~~~~~~~

.. autoapisummary::

   torchtree.inference.hmc.integrator.set_tensor



.. py:function:: set_tensor(parameters, tensor: torch.Tensor) -> None


.. py:class:: Integrator(id_)



   Helper class that provides a standard way to create an ABC using
   inheritance.


.. py:class:: LeapfrogIntegrator(id_, steps: int, step_size: float)



   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: from_json(data, dic)
      :classmethod:



