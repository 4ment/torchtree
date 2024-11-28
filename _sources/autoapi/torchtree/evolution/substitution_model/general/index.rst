torchtree.evolution.substitution_model.general
==============================================

.. py:module:: torchtree.evolution.substitution_model.general


Classes
-------

.. autoapisummary::

   torchtree.evolution.substitution_model.general.GeneralJC69
   torchtree.evolution.substitution_model.general.GeneralSymmetricSubstitutionModel
   torchtree.evolution.substitution_model.general.GeneralNonSymmetricSubstitutionModel
   torchtree.evolution.substitution_model.general.EmpiricalSubstitutionModel


Module Contents
---------------

.. py:class:: GeneralJC69(id_: torchtree.typing.ID, state_count: int)

   Bases: :py:obj:`torchtree.evolution.substitution_model.abstract.SubstitutionModel`


   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.


   .. py:attribute:: state_count


   .. py:property:: frequencies
      :type: torch.Tensor



   .. py:property:: rates
      :type: Union[torch.Tensor, list[torch.Tensor]]



   .. py:method:: handle_model_changed(model, obj, index)


   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: cuda(device: Optional[Union[int, torch.device]] = None) -> None

      Move tensors to CUDA using torch.cuda.



   .. py:method:: cpu() -> None

      Move tensors to CPU memory using ~torch.cpu.



   .. py:method:: p_t(branch_lengths: torch.Tensor) -> torch.Tensor


   .. py:method:: q() -> torch.Tensor


   .. py:method:: from_json(data, dic)
      :classmethod:


      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: GeneralSymmetricSubstitutionModel(id_: torchtree.typing.ID, data_type: torchtree.evolution.datatype.DataType, mapping: torchtree.core.abstractparameter.AbstractParameter, rates: torchtree.core.abstractparameter.AbstractParameter, frequencies: torchtree.core.abstractparameter.AbstractParameter)

   Bases: :py:obj:`torchtree.evolution.substitution_model.abstract.SymmetricSubstitutionModel`


   General symmetric substitution model.

   The state space :math:`\Omega=\{S_0, S_1, \dots, S_{M-1}\}` of this model is defined by the `DataType` object.

   This model is composed of:

   - :math:`K` substitution rate parameters: :math:`\mathbf{r}=r_0, r_1, \dots, r_{K-1}` where :math:`K \leq (M^2-M)/2`.
   - :math:`M` equilibrium frequency parameters: :math:`\pi_0, \pi_1, \dots, \pi_{M-1}`.
   - A mapping function that associates each matrix element :math:`Q_{ij}` to an index in the set of rates :math:`f: \{0, 1, \dots, (M^2-M)/2-1\} \rightarrow \{0,1, \dots, K-1\}`

   The matrix :math:`Q` is thus defined as:

   .. math::

       Q_{ij} =
       \begin{cases}
       r_{f(i \cdot M + j)} \pi_j & \text{if } i \neq j \\
       -\sum_{k \neq i} Q_{ik} & \text{if } i = j
       \end{cases}

   where :math:`i,j \in \{0,1, \dots, M-1\}` are zero-based indices for rows and columns.

   :math:`f` is implemented as a one-dimentional array :math:`\mathbf{g}[x]=f(x)` for :math:`x \in \{0, 1,\dots, (M^2-M)/2-1\}` where each element maps a position in :math:`Q` to an index in the rate array :math:`r`.
   The mapping is defined such as the position :math:`(i,j)` in :math:`Q` corresponds to :math:`i \cdot M+ j` for :math:`i \neq j`.
   The indices correspond to first iterating over rows (row 0, then row 1, etc.) and then over columns for each row of the upper off-diagonal elements.

   The HKY substitution model can be defined as a symmetric substitution model with M=4 frequency parameters and rate parameters :math:`\mathbf{r}=r_0, r_1`.
   The mapping function is therefore:

   .. math::

       f(k) =
       \begin{cases}
       0 & \text{if } k = i \cdot 4 + j \text{ and } i \rightarrow j \text{ is transversion}\\
       1 & \text{otherwise}
       \end{cases}

   As a one-dimentional array, the mapping is defined as :math:`\mathbf{g}=[0,1,0,0,1,0]`.

   The HKY rate matrix :math:`Q` is given as:

   .. math::

       Q_{HKY} =
       \begin{bmatrix}
       -(r_0 \pi_C + r_1 \pi_G + r_0 \pi_T) & r_0 \pi_C & r_1 \pi_G & r_0 \pi_T \\
       r_0 \pi_A & -(r_0 \pi_A + r_0 \pi_G + r_0 \pi_T) & r_0 \pi_G & r_1 \pi_T \\
       r_1 \pi_A & r_0 \pi_C & -(r_1\pi_A + r_0 \pi_C + r_0 \pi_T) & r_0 \pi_T \\
       r_0 \pi_A & r_1 \pi_C & r_0 \pi_G & -(r_0 \pi_A + r_1 \pi_C + r_0 \pi_G)
       \end{bmatrix}

   Similarly the GTR model can be specified with :math:`\mathbf{g}=[0,1,2,3,4,5]` and :math:`\mathbf{r}=r_0, r_1, r_2, r_3, r_4, r_5`.

   .. note::
       The order of the equilibrium frequencies in a :class:`~torchtree.Parameter` is expected to be the order of the states defined in the DataType object.


   .. py:attribute:: mapping


   .. py:attribute:: state_count


   .. py:attribute:: data_type


   .. py:property:: rates
      :type: Union[torch.Tensor, list[torch.Tensor]]



   .. py:method:: handle_model_changed(model, obj, index)


   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: q() -> torch.Tensor


   .. py:method:: from_json(data, dic)
      :classmethod:


      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: GeneralNonSymmetricSubstitutionModel(id_: torchtree.typing.ID, data_type: torchtree.evolution.datatype.DataType, mapping: torchtree.core.abstractparameter.AbstractParameter, rates: torchtree.core.abstractparameter.AbstractParameter, frequencies: torchtree.core.abstractparameter.AbstractParameter, normalize: bool)

   Bases: :py:obj:`torchtree.evolution.substitution_model.abstract.NonSymmetricSubstitutionModel`


   General non-symmetric substitution model.

   The state space :math:`\Omega=\{S_0, S_1, \dots, S_{M-1}\}` of this model is defined by the `DataType` object.

   This model is composed of:

   - :math:`K` substitution rate parameters: :math:`\mathbf{r}=r_0, r_1, \dots, r_{K-1}` where :math:`K \leq (M^2-M)`.
   - :math:`M` equilibrium frequency parameters: :math:`\pi_0, \pi_1, \dots, \pi_{M-1}`.
   - A mapping function that associates each matrix element :math:`Q_{ij}` to an index in the set of rates :math:`f: \{0, 1, \dots, (M^2-M)-1\} \rightarrow \{0,1, \dots, K-1\}`

   The matrix :math:`Q` is thus defined as:

   .. math::

       Q_{ij} =
       \begin{cases}
       r_{f(i \cdot M + j)} \pi_j & \text{if } i \neq j \\
       -\sum_{k \neq i} Q_{ik} & \text{if } i = j
       \end{cases}

   where :math:`i,j \in \{0,1, \dots, M-1\}` are zero-based indices for rows and columns.

   :math:`f` is implemented as a one-dimentional array :math:`\mathbf{g}[x]=f(x)` for :math:`x \in \{0, 1,\dots, (M^2-M)-1\}` where each element maps a position in :math:`Q` to an index in the rate array :math:`r`.
   The mapping is defined such as the position :math:`(i,j)` in :math:`Q` corresponds to :math:`i \cdot M + j` for :math:`i > j` and :math:`j \cdot M + i + (M^2-M)/2` for :math:`i < j`.
   In other words, the first of :math:`\mathbf{g}` corresponds to the upper off-diagonal elements and the second to the lower off-diagonal elements.

   .. note::
       The order of the equilibrium frequencies in a :class:`~torchtree.Parameter` is expected to be the order of the states defined in the DataType object.


   .. py:attribute:: mapping


   .. py:attribute:: state_count


   .. py:attribute:: data_type


   .. py:attribute:: normalize


   .. py:property:: rates
      :type: torch.Tensor



   .. py:method:: handle_model_changed(model, obj, index)


   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: q() -> torch.Tensor


   .. py:method:: from_json(data, dic)
      :classmethod:


      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: EmpiricalSubstitutionModel(id_: torchtree.typing.ID, rates: torch.Tensor, frequencies: torch.Tensor)

   Bases: :py:obj:`torchtree.evolution.substitution_model.abstract.SubstitutionModel`


   Parametric model.

   A Model can contain parameters and models and can monitor any
   changes. A Model is the building block of more complex models. This
   class is abstract.


   .. py:attribute:: Q


   .. py:attribute:: sqrt_pi


   .. py:attribute:: sqrt_pi_inv


   .. py:property:: frequencies
      :type: torch.Tensor



   .. py:method:: q() -> torch.Tensor


   .. py:method:: p_t(branch_lengths: torch.Tensor) -> torch.Tensor


   .. py:method:: eigen(Q: torch.Tensor) -> torch.Tensor


   .. py:method:: handle_model_changed(model, obj, index) -> None


   .. py:method:: handle_parameter_changed(variable: torchtree.core.abstractparameter.AbstractParameter, index, event) -> None


   .. py:method:: create_rate_matrix(rates: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor
      :staticmethod:



   .. py:method:: from_json(data, dic)
      :classmethod:


      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



