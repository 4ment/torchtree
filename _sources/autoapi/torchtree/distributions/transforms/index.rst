:py:mod:`torchtree.distributions.transforms`
============================================

.. py:module:: torchtree.distributions.transforms

.. autoapi-nested-parse::

   Invertible transformations inheriting from torch.distributions.Transform.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.distributions.transforms.TrilExpDiagonalTransform
   torchtree.distributions.transforms.CumSumTransform
   torchtree.distributions.transforms.CumSumExpTransform
   torchtree.distributions.transforms.SoftPlusTransform
   torchtree.distributions.transforms.CumSumSoftPlusTransform
   torchtree.distributions.transforms.ConvexCombinationTransform
   torchtree.distributions.transforms.LogTransform
   torchtree.distributions.transforms.LinearTransform




.. py:class:: TrilExpDiagonalTransform(cache_size=0)


   Bases: :py:obj:`torch.distributions.Transform`

   Transform a 1D tensor to a triangular tensor.

   The diagonal of the triangular matrix is exponentiated. Useful for variational
   inference with the multivariate normal distribution as the variational
   distribution and it is parameterized with scale_tril, a lower-triangular matrix
   with positive diagonal.

   .. rubric:: Example

   >>> x = torch.tensor([1., 2., 3.])
   >>> y = TrilExpDiagonalTransform()(x)
   >>> y
   tensor([[ 2.7183,  0.0000],
           [ 2.0000, 20.0855]])
   >>> torch.allclose(TrilExpDiagonalTransform().inv(y), x)
   True

   .. py:attribute:: bijective
      :value: True

      

   .. py:attribute:: sign

      

   .. py:method:: log_abs_det_jacobian(x, y)
      :abstractmethod:

      Computes the log det jacobian `log |dy/dx|` given input and output.



.. py:class:: CumSumTransform(cache_size=0)


   Bases: :py:obj:`torch.distributions.Transform`

   Transform via the mapping :math:`y_i = \sum_{j=0}^i x_j`.

   >>> x = torch.tensor([1., 2., 3.])
   >>> all(CumSumTransform()(x) == torch.tensor([1., 3., 6.]))
   True
   >>> all(CumSumTransform().inv(torch.tensor([1., 3., 6.])) == x)
   True

   .. py:attribute:: domain

      

   .. py:attribute:: codomain

      

   .. py:attribute:: bijective
      :value: True

      

   .. py:attribute:: sign

      

   .. py:method:: log_abs_det_jacobian(x, y)

      Computes the log det jacobian `log |dy/dx|` given input and output.



.. py:class:: CumSumExpTransform(cache_size=0)


   Bases: :py:obj:`torch.distributions.Transform`

   Transform via the mapping :math:`y_i = \exp(\sum_{j=0}^i x_j)`.

   .. py:attribute:: domain

      

   .. py:attribute:: codomain

      

   .. py:attribute:: bijective
      :value: True

      

   .. py:attribute:: sign

      

   .. py:method:: log_abs_det_jacobian(x, y)

      Computes the log det jacobian `log |dy/dx|` given input and output.



.. py:class:: SoftPlusTransform(cache_size=0)


   Bases: :py:obj:`torch.distributions.Transform`

   Transform via the mapping :math:`y_i = \log(\exp(x_i) + 1)`.

   .. py:attribute:: domain

      

   .. py:attribute:: codomain

      

   .. py:attribute:: bijective
      :value: True

      

   .. py:attribute:: sign

      

   .. py:method:: log_abs_det_jacobian(x, y)

      Computes the log det jacobian `log |dy/dx|` given input and output.



.. py:class:: CumSumSoftPlusTransform(cache_size=0)


   Bases: :py:obj:`torch.distributions.Transform`

   Transform via the mapping :math:`y_i = \log(\exp(\sum_{j=0}^i x_j) +1)`.

   .. py:attribute:: domain

      

   .. py:attribute:: codomain

      

   .. py:attribute:: bijective
      :value: True

      

   .. py:attribute:: sign

      

   .. py:method:: log_abs_det_jacobian(x, y)

      Computes the log det jacobian `log |dy/dx|` given input and output.



.. py:class:: ConvexCombinationTransform(weights: torchtree.core.abstractparameter.AbstractParameter, cache_size=0)


   Bases: :py:obj:`torch.distributions.Transform`

   Transform via the mapping :math:`y = \frac{x}{\sum_{i=1}^K \alpha_i x_i}`.

   The transformation satisfies
   :math:`\sum_{i=1}^K \alpha_i y_i = 1` where :math:`\alpha_i \geq 0` and
   :math:`\sum_{i=1}^K \alpha_i = 1`.

   :param weights: weights (sum to 1)

   .. py:attribute:: domain

      

   .. py:attribute:: codomain

      

   .. py:method:: log_abs_det_jacobian(x, y)

      Computes the log det jacobian `log |dy/dx|` given input and output.



.. py:class:: LogTransform(cache_size=0)


   Bases: :py:obj:`torch.distributions.Transform`

   Transform via the mapping :math:`y = \log(x)`.

   .. py:attribute:: domain

      

   .. py:attribute:: codomain

      

   .. py:attribute:: bijective
      :value: True

      

   .. py:attribute:: sign

      

   .. py:method:: log_abs_det_jacobian(x, y)

      Computes the log det jacobian `log |dy/dx|` given input and output.



.. py:class:: LinearTransform(weight: Union[torchtree.core.abstractparameter.AbstractParameter, torch.Tensor], bias: Optional[Union[torchtree.core.abstractparameter.AbstractParameter, torch.Tensor]] = None, cache_size=0)


   Bases: :py:obj:`torch.distributions.Transform`

   Transform via the mapping :math:`y = xA' + b`.

   :example:
   >>> x = torch.rand(3,2)
   >>> weight = torch.rand(5, 2)
   >>> bias = torch.rand(5)
   >>> transform = LinearTransform(weight, bias)
   >>> y = transform(x)
   >>> torch.all(y == torch.nn.functional.linear(x, weight, bias))
   tensor(True)

   .. py:attribute:: domain

      

   .. py:attribute:: codomain

      


