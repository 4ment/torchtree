:py:mod:`torchtree.distributions.ctmc_scale`
============================================

.. py:module:: torchtree.distributions.ctmc_scale


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.distributions.ctmc_scale.CTMCScale




.. py:class:: CTMCScale(id_: torchtree.typing.ID, x: torchtree.core.abstractparameter.AbstractParameter, tree_model: torchtree.evolution.tree_model.TreeModel)



   Class implementing the CTMC scale prior [#ferreira2008]_

   :param id_: ID of object
   :type id_: str or None
   :param torch.Tensor x: substitutin rate parameter
   :param TreeModel tree_model: tree model

   .. [#ferreira2008] Ferreira and Suchard. Bayesian analysis of elapsed times
    in continuous-time Markov chains. 2008

   .. py:attribute:: shape

      

   .. py:attribute:: log_gamma_one_half

      

   .. py:method:: to(*args, **kwargs) -> None


   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None) -> None


   .. py:method:: cpu() -> None


   .. py:method:: json_factory(id_: str, rate: Union[str, dict], tree: Union[str, dict]) -> dict
      :staticmethod:


   .. py:method:: from_json(data, dic)
      :classmethod:



