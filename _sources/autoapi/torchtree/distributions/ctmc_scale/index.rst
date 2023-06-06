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


   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None) -> None


   .. py:method:: cpu() -> None


   .. py:method:: json_factory(id_: str, rate: Union[str, dict], tree: Union[str, dict]) -> dict
      :staticmethod:


   .. py:method:: from_json(data, dic)
      :classmethod:



