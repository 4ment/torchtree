:py:mod:`torchtree.evolution.site_pattern`
==========================================

.. py:module:: torchtree.evolution.site_pattern


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.evolution.site_pattern.SitePattern



Functions
~~~~~~~~~

.. autoapisummary::

   torchtree.evolution.site_pattern.compress
   torchtree.evolution.site_pattern.compress_alignment
   torchtree.evolution.site_pattern.compress_alignment_states



.. py:class:: SitePattern(id_: Optional[str], alignment: torchtree.evolution.alignment.Alignment, indices: list[Union[int, slice]] = None)


   Bases: :py:obj:`torchtree.core.model.Model`

   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.

   .. py:method:: compute_tips_partials(use_ambiguities=False)


   .. py:method:: compute_tips_states()


   .. py:method:: handle_model_changed(model, obj, index)


   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None) -> None

      Move tensors to CUDA using torch.cuda.


   .. py:method:: cpu() -> None

      Move tensors to CPU memory using ~torch.cpu.


   .. py:method:: from_json(data, dic)
      :classmethod:

      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:function:: compress(alignment: torchtree.evolution.alignment.Alignment, indices: list[Union[int, slice]] = None) -> tuple[dict[str, tuple[str]], torch.Tensor]

   Compress alignment using data_type.

   :param Alignment alignment: sequence alignment
   :param indices: list of indices: int or slice
   :return: a tuple containing partials and weights
   :rtype: Tuple[Dict[str, Tuple[str]], torch.Tensor]


.. py:function:: compress_alignment(alignment: torchtree.evolution.alignment.Alignment, indices: list[Union[int, slice]] = None, use_ambiguities=True) -> tuple[list[torch.Tensor], torch.Tensor]

   Compress alignment using data_type.

   :param Alignment alignment: sequence alignment
   :param indices: list of indices: int or slice
   :return: a tuple containing partials and weights
   :rtype: Tuple[List[torch.Tensor], torch.Tensor]


.. py:function:: compress_alignment_states(alignment: torchtree.evolution.alignment.Alignment, indices: list[Union[int, slice]] = None) -> tuple[list[torch.Tensor], torch.Tensor]

   Compress alignment using data_type.

   :param Alignment alignment: sequence alignment
   :param indices: list of indices: int or slice
   :return: a tuple containing partials and weights
   :rtype: Tuple[List[torch.Tensor], torch.Tensor]


