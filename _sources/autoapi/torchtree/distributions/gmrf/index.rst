:py:mod:`torchtree.distributions.gmrf`
======================================

.. py:module:: torchtree.distributions.gmrf

.. autoapi-nested-parse::

   Gaussian Markov random field priors.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.distributions.gmrf.GMRF
   torchtree.distributions.gmrf.GMRFCovariate




.. py:class:: GMRF(id_: torchtree.typing.ID, field: torchtree.core.abstractparameter.AbstractParameter, precision: torchtree.core.abstractparameter.AbstractParameter, tree_model: torchtree.evolution.tree_model.TimeTreeModel = None, weights: torch.Tensor = None, rescale: bool = True)


   Bases: :py:obj:`torchtree.core.model.CallableModel`

   Gaussian Markov random field.

   GMRF is parameterized with precision :math:`\tau` parameter.

   :param id_: ID of GMRF object.
   :type id_: str or None
   :param AbstractParameter field: Markov random field parameter.
   :param AbstractParameter precision: precision parameter.
   :param TimeTreeModel tree_model: Optional; time tree model.
       (if specified a time-aware GMRF is used).
   :param bool rescale: Optional; rescale by root height
       (tree_model must be specified).

   .. math::
      p(\boldsymbol{x} \mid \tau) =  \prod_{i=1}^{N-1} \frac{1}{\sqrt{2 \pi}}
       \sqrt{\tau} e^{-\frac{\tau}{2} (x_{i+1} -x_i)^2}

   .. py:method:: from_json(data: dict[str, Any], dic: dict[str, torchtree.core.identifiable.Identifiable]) -> GMRF
      :classmethod:

      Creates a GMRF object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a GMRF object.
      :param dict[str, Identifiable] dic: dictionary containing torchtree objects
          keyed by their ID.

      **JSON attributes**:

       Mandatory:
        - id (str): unique string identifier.
        - x (dict or str): Markov random field parameter.
        - precision (dict or str): precision parameter.

       Optional:
        - tree_model (dict or str):time tree model.
        - rescale (bool): rescale by root height (Default: true).

      :example:
      >>> field = {"id": "field", "type": "Parameter", "tensor": [1., 2., 3.]}
      >>> precision = {"id": "precision", "type": "Parameter", "tensor": [1.]}
      >>> gmrf_dic = {"id": "gmrf", "x": field, "precision", precision}
      >>> gmrf = GMRF.from_json(gmrf_dic, {})
      >>> isinstance(gmrf, GMRF)
      True

      .. note::
          If tree_model is specified the GMRF is time-aware and it should not be used
          with skygrid. The rescale parameter is ignored if tree_model is not
          specified.



.. py:class:: GMRFCovariate(id_: torchtree.typing.ID, field: torchtree.core.abstractparameter.AbstractParameter, precision: torchtree.core.abstractparameter.AbstractParameter, covariates: torchtree.core.abstractparameter.AbstractParameter, beta: torchtree.core.abstractparameter.AbstractParameter)


   Bases: :py:obj:`torchtree.core.model.CallableModel`

   Gaussian Markov random field with covariates.

   Creates the Gaussian Markov random field with covariates prior proposed
   by\ :footcite:t:`gill2016understanding`.

   :param id_: ID of GMRF object.
   :type id_: str or None
   :param AbstractParameter field: Markov random field.
   :param AbstractParameter precision: precision parameter.
   :param AbstractParameter covariates: covariates.
   :param AbstractParameter beta: coefficients representing the effect sizes for the
       covariates.

   Let :math:`Z_{1}, \ldots , Z_{P}` be a set of :math:`\boldsymbol{Z}` predictors.
   :math:`Z_i` is observed or measured at N time points.
   :math:`x_i` is as a linear function of covariates

   .. math::
       x_i = \sum \beta_{ip} Z_{ip} + w_i

   where :math:`\boldsymbol{w}=(w_1 \ldots w_N)` is a zero-mean Gaussian process and
   :math:`\boldsymbol{\beta}=(\beta_1 \ldots \beta_N)` are coefficients.

   .. math::
       p(\boldsymbol{x} \mid \boldsymbol{Z}, \boldsymbol{\beta}, \tau)
       \propto \tau^{(N-1)/2}  e^{-\tau/2(X - \boldsymbol{Z} \boldsymbol{\beta})'
       \boldsymbol{Q} (X - \boldsymbol{Z} \boldsymbol{\beta})}

   .. footbibliography::

   .. py:method:: from_json(data: dict[str, Any], dic: dict[str, torchtree.core.identifiable.Identifiable]) -> GMRFCovariate
      :classmethod:

      Creates a GMRFCovariate object from a dictionary.

      :param dict[str, Any] data: dictionary representation of a GMRFCovariate
          object.
      :param dict[str, Identifiable] dic: dictionary containing torchtree objects
          keyed by their ID.

      **JSON attributes**:

       Mandatory:
        - id (str): unique string identifier.
        - x (dict or str): Markov random field parameter.
        - precision (dict or str): precision parameter.
        - covariates (dict or str or list): covariates.
        - beta (dict or str): coefficients.

      .. note::
          If the shape of the field parameter is [...,N] and there are P covariates
          then the shape of the covariates parameter should be [N,P] and the shape
          of the beta parameter should be [...,P].



