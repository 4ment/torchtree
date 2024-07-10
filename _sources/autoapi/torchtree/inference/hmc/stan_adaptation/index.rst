torchtree.inference.hmc.stan_adaptation
=======================================

.. py:module:: torchtree.inference.hmc.stan_adaptation


Classes
-------

.. autoapisummary::

   torchtree.inference.hmc.stan_adaptation.StanWindowedAdaptation


Module Contents
---------------

.. py:class:: StanWindowedAdaptation(step_size_adaptor: torchtree.inference.hmc.adaptation.DualAveragingStepSize, mass_matrix_adaptor: torchtree.inference.hmc.adaptation.Adaptor, num_warmup: int, init_buffer: int, term_buffer: int, base_window: int)

   Bases: :py:obj:`torchtree.inference.hmc.adaptation.Adaptor`


   Adapts step size and mass matrix during a warmup period.

   Code adapted from Stan. See online manual for further details
   https://mc-stan.org/docs/reference-manual/hmc-algorithm-parameters.html

   :param step_size_adaptor: step size adaptor
   :param mass_matrix_adaptor: mass matrix adaptor
   :param int num_warmup: number of iteration of warmup period
   :param int init_buffer: width of initial fast adaptation interval
   :param int term_buffer: width of final fast adaptation interval
   :param int base window: initial width of slow adaptation interval


   .. py:method:: restart()


   .. py:method:: learn(acceptance_prob: torch.Tensor, sample: int, accepted: bool) -> None


   .. py:method:: from_json(data, dic)
      :classmethod:


      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



