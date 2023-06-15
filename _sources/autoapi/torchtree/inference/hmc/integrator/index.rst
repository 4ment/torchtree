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


   Bases: :py:obj:`torchtree.core.identifiable.Identifiable`, :py:obj:`abc.ABC`

   Abstract class making an object identifiable.

   :param str or None id_: identifier of object


.. py:class:: LeapfrogIntegrator(id_, steps: int, step_size: float)


   Bases: :py:obj:`Integrator`

   Abstract class making an object identifiable.

   :param str or None id_: identifier of object

   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



