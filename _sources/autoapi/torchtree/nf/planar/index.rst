:py:mod:`torchtree.nf.planar`
=============================

.. py:module:: torchtree.nf.planar


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.nf.planar.PlanarTransform




.. py:class:: PlanarTransform(u: torch.nn.Parameter, w: torch.nn.Parameter, b: torch.nn.Parameter)


   Bases: :py:obj:`torch.nn.Module`

   Implementation of the transformation used in planar flow:

   f(z) = z + u * tanh(dot(w.T, z) + b)

   where z are the inputs and u, w, and b are learnable parameters.
   The shape of z is (batch_size, input_size).

   :param Parameter u: scaling factor with shape(1, input_size)
   :param Parameter w: weight with shape (1, input_size)
   :param Parameter b: bias with shape (1)

   .. py:method:: forward(z: torch.Tensor) -> torch.Tensor


   .. py:method:: log_abs_det_jacobian(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor


   .. py:method:: u_hat() -> torch.Tensor



