:py:mod:`torchtree.nf.realnvp`
==============================

.. py:module:: torchtree.nf.realnvp

.. autoapi-nested-parse::

   Masked Autoregressive Flow for Density Estimation arXiv:1705.07057v4 Code
   ported from https://github.com/kamenbliznashki/normalizing_flows.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.nf.realnvp.LinearMaskedCoupling
   torchtree.nf.realnvp.BatchNorm
   torchtree.nf.realnvp.FlowSequential
   torchtree.nf.realnvp.RealNVP




.. py:class:: LinearMaskedCoupling(input_size, hidden_size, n_hidden, mask, cond_label_size=None)


   Bases: :py:obj:`torch.nn.Module`

   Modified RealNVP Coupling Layers per the MAF paper.

   .. py:method:: forward(x, y=None)


   .. py:method:: inverse(u, y=None)



.. py:class:: BatchNorm(input_size, momentum=0.9, eps=1e-05)


   Bases: :py:obj:`torch.nn.Module`

   RealNVP BatchNorm layer.

   .. py:method:: forward(x, cond_y=None)


   .. py:method:: inverse(y, cond_y=None)



.. py:class:: FlowSequential(*args: torch.nn.modules.module.Module)
              FlowSequential(arg: OrderedDict[str, Module])


   Bases: :py:obj:`torch.nn.Sequential`

   Container for layers of a normalizing flow.

   .. py:method:: forward(x, y)


   .. py:method:: inverse(u, y)



.. py:class:: RealNVP(id_: str, x: torchtree.core.abstractparameter.AbstractParameter, base: torchtree.distributions.distributions.Distribution, n_blocks: int, hidden_size: int, n_hidden: int, cond_label_size=None, batch_norm=False)


   Bases: :py:obj:`torchtree.distributions.distributions.DistributionModel`

   Class for RealNVP normalizing flows.

   :param id_: ID of object
   :param x: parameter or list of parameters
   :param base: base distribution
   :param n_blocks:
   :param hidden_size:
   :param n_hidden:
   :param cond_label_size:
   :param batch_norm:

   .. py:property:: batch_shape
      :type: torch.Size


   .. py:method:: forward(x, y=None)


   .. py:method:: inverse(u, y=None)


   .. py:method:: apply_flow(sample_shape: torch.Size)


   .. py:method:: sample(sample_shape=torch.Size()) -> None

      Generates a sample_shape shaped sample or sample_shape shaped batch
      of samples if the distribution parameters are batched.


   .. py:method:: rsample(sample_shape=torch.Size()) -> None

      Generates a sample_shape shaped reparameterized sample or
      sample_shape shaped batch of reparameterized samples if the
      distribution parameters are batched.


   .. py:method:: log_prob(x: torchtree.core.abstractparameter.AbstractParameter = None) -> torch.Tensor

      Returns the log of the probability density/mass function evaluated at x.

      :param Parameter x: value to evaluate
      :return: log probability
      :rtype: Tensor


   .. py:method:: parameters() -> list[torchtree.core.abstractparameter.AbstractParameter]

      Returns parameters of instance Parameter.


   .. py:method:: entropy() -> torch.Tensor

      Returns entropy of distribution, batched over batch_shape.

      :return: Tensor of shape batch_shape.
      :rtype: Tensor


   .. py:method:: from_json(data, dic) -> RealNVP
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



