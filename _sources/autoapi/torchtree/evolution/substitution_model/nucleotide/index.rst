torchtree.evolution.substitution_model.nucleotide
=================================================

.. py:module:: torchtree.evolution.substitution_model.nucleotide

.. autoapi-nested-parse::

   Reversible nucleotide substitution models.

   Reversible nucleotide substitution models are characterized by the following:

   1. **Time Reversibility**

   The substitution process satisfies the detailed balance condition:

   .. math::

       \pi_i Q_{ij} = \pi_j Q_{ji}

   where:

   - :math:`\pi_i` and :math:`\pi_j` are the equilibrium frequencies of nucleotides :math:`i` and :math:`j`.
   - :math:`Q_{ij}` is the rate of substitution from nucleotide :math:`i` to :math:`j`.

   This ensures that the process appears the same forward and backward in time, making the likelihood computations simpler.

   2. **Equilibrium Frequencies**

   Represent the long-term stationary distribution of nucleotides. These frequencies account for biases in nucleotide composition.

   3. **Rate Matrix (Q)**

   The general structure of the rate matrix for reversible models is:

   .. math::

       Q_{ij} =
       \begin{cases}
       \mu_{ij} \pi_j & \text{if } i \neq j \\
       -\sum_{k \neq i} Q_{ik} & \text{if } i = j
       \end{cases}

   where:

   - :math:`\mu_{ij}` is the relative rate of substitution between nucleotides :math:`i` and :math:`j`.
   - Diagonal entries (:math:`Q_{ii}`) ensure rows sum to zero.

   4. **Scaling**

   The rate matrix is typically scaled such that the average rate of substitution is 1.
   The scaling factor :math:`\beta` is given by:

   .. math::
       \beta = -\frac{1}{\sum_{i} \pi_i \mu_{ii}}

   .. note::
       The order of the equilibrium frequencies in a :class:`~torchtree.Parameter` is expected to be :math:`\pi_A, \pi_C, \pi_G, \pi_T`.



Classes
-------

.. autoapisummary::

   torchtree.evolution.substitution_model.nucleotide.JC69
   torchtree.evolution.substitution_model.nucleotide.HKY
   torchtree.evolution.substitution_model.nucleotide.GTR


Module Contents
---------------

.. py:class:: JC69(id_: torchtree.typing.ID)

   Bases: :py:obj:`torchtree.evolution.substitution_model.abstract.SubstitutionModel`


   Jukes-Cantor (JC69) substitution model.

   The JC69 model assumes:

   - **Equal substitution rates:** Each nucleotide is equally likely to mutate into another nucleotide.
   - **Equal base frequencies:** The equilibrium frequencies of :math:`\pi_A, \pi_C, \pi_G, \pi_T` are all equal to 0.25.
   - **Reversibility:** The substitution process is time-reversible.

   The JC69 rate matrix :math:`Q` is given as:

   .. math::

       Q =
       \begin{bmatrix}
       -1 & 1/3 & 1/3 & 1/3 \\
       1/3 & -1 & 1/3 & 1/3 \\
       1/3 & 1/3 & -1 & 1/3 \\
       1/3 & 1/3 & 1/3 & -1
       \end{bmatrix}



   .. py:property:: frequencies
      :type: torch.Tensor



   .. py:property:: rates
      :type: Union[torch.Tensor, list[torch.Tensor]]



   .. py:method:: p_t(branch_lengths: torch.Tensor) -> torch.Tensor

      Calculate transition probability matrices.

      :param branch_lengths: tensor of branch lengths [B,K]
      :return: tensor of probability matrices [B,K,4,4]



   .. py:method:: q() -> torch.Tensor


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



.. py:class:: HKY(id_: torchtree.typing.ID, kappa: torchtree.core.abstractparameter.AbstractParameter, frequencies: torchtree.core.abstractparameter.AbstractParameter)

   Bases: :py:obj:`torchtree.evolution.substitution_model.abstract.SymmetricSubstitutionModel`


   Hasegawa-Kishino-Yano (HKY) substitution model.

   The HKY model has:

   - A transition/transversion rate ratio parameters: :math:`\kappa`.
   - Four equilibrium frequency parameters: :math:`\pi_A, \pi_C, \pi_G, \pi_T`.

   The HKY rate matrix :math:`Q` is given as:

   .. math::

       Q =
       \begin{bmatrix}
       -(\pi_C + \kappa \pi_G + \pi_T) & \pi_C & \kappa \pi_G & \pi_T \\
       \pi_A & -(\pi_A + \pi_G + \kappa \pi_T) & \pi_G & \kappa \pi_T \\
       \kappa \pi_A & \pi_C & -(\kappa \pi_A + \pi_C + \pi_T) & \pi_T \\
       \pi_A & \kappa \pi_C & \pi_G & -(\pi_A + \kappa \pi_C + \pi_G)
       \end{bmatrix}


   .. py:property:: rates
      :type: Union[torch.Tensor, list[torch.Tensor]]



   .. py:property:: kappa
      :type: torch.Tensor



   .. py:method:: handle_model_changed(model, obj, index)


   .. py:method:: handle_parameter_changed(variable, index, event)


   .. py:method:: p_t_analytical(branch_lengths: torch.Tensor) -> torch.Tensor
      :abstractmethod:



   .. py:method:: q() -> torch.Tensor


   .. py:method:: from_json(data, dic)
      :classmethod:


      Abstract method to create object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a torchtree object.
      :param dict[str, Any] dic: dictionary containing other torchtree objects keyed
          by their ID.
      :return: torchtree object.
      :rtype: Any



.. py:class:: GTR(id_: torchtree.typing.ID, rates: torchtree.core.abstractparameter.AbstractParameter, frequencies: torchtree.core.abstractparameter.AbstractParameter)

   Bases: :py:obj:`torchtree.evolution.substitution_model.abstract.SymmetricSubstitutionModel`


   General Time Reversible (GTR) substitution model.

   The GTR model has:

   - Six substitution rate parameters: :math:`a, b, c, d, e, f`.
   - Four equilibrium frequency parameters: :math:`\pi_A, \pi_C, \pi_G, \pi_T`.

   The GTR rate matrix :math:`Q` is given as:

   .. math::

       Q =
       \begin{bmatrix}
       -(a \pi_C + b \pi_G + c \pi_T) & a \pi_C & b \pi_G & c \pi_T \\
       a \pi_A & -(a \pi_A + d \pi_G + e \pi_T) & d \pi_G & e \pi_T \\
       b \pi_A & d \pi_C & -(b \pi_A + d \pi_C + f \pi_T) & f \pi_T \\
       c \pi_A & e \pi_C & f \pi_G & -(c \pi_A + e \pi_C + f \pi_G)
       \end{bmatrix}

   where the exchangeability parameters are defined as:

   .. math::
       \begin{align*}
       a &= r_{AC} = r_{CA}\\
       b &= r_{AG} = r_{GA}\\
       c &= r_{AT} = r_{TA}\\
       d &= r_{CG} = r_{GC}\\
       e &= r_{CT} = r_{TC}\\
       f &= r_{GT} = r_{TG}
       \end{align*}

   .. note::
       The order of the rate parameters in a :class:`~torchtree.Parameter` is expected to be :math:`a, b, c, d, e, f`.
       The upper off-diagonal elements are indexed by first iterating over rows (row 0, then row 1, etc.) and then over columns for each row.


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



