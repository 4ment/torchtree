:py:mod:`torchtree.distributions.ctmc_scale`
============================================

.. py:module:: torchtree.distributions.ctmc_scale

.. autoapi-nested-parse::

   CTMC scale reference prior.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.distributions.ctmc_scale.CTMCScale




.. py:class:: CTMCScale(id_: torchtree.typing.ID, x: torchtree.core.abstractparameter.AbstractParameter, tree_model: torchtree.evolution.tree_model.TreeModel)


   Bases: :py:obj:`torchtree.core.model.CallableModel`

   Continuous-time Markov chain scale prior.

   Class implementing the CTMC reference scale prior from
   :footcite:t:`ferreira2008bayesian`.

   :param id_: ID of object
   :type id_: str or None
   :param torch.Tensor x: substitution rate parameter
   :param TreeModel tree_model: tree model

   .. footbibliography::

   .. py:attribute:: shape

      

   .. py:attribute:: log_gamma_one_half

      

   .. py:method:: to(*args, **kwargs) -> None

      Performs Tensor dtype and/or device conversion using torch.to.


   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None) -> None

      Move tensors to CUDA using torch.cuda.


   .. py:method:: cpu() -> None

      Move tensors to CPU memory using ~torch.cpu.


   .. py:method:: json_factory(id_: str, rate: Union[str, dict], tree: Union[str, dict]) -> dict
      :staticmethod:


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



