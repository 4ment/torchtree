torchtree.cli.advi
==================

.. py:module:: torchtree.cli.advi


Attributes
----------

.. autoapisummary::

   torchtree.cli.advi.logger


Functions
---------

.. autoapisummary::

   torchtree.cli.advi.create_variational_parser
   torchtree.cli.advi.create_tril
   torchtree.cli.advi.create_fullrank_from_meanfield
   torchtree.cli.advi.create_fullrank
   torchtree.cli.advi.create_flexible_variational
   torchtree.cli.advi.create_realnp_distribution
   torchtree.cli.advi.create_realnvp
   torchtree.cli.advi.gather_parameters
   torchtree.cli.advi.apply_sigmoid_transformed
   torchtree.cli.advi.apply_affine_transform
   torchtree.cli.advi.apply_exp_transform
   torchtree.cli.advi.apply_simplex_transform
   torchtree.cli.advi.create_normal_distribution
   torchtree.cli.advi.create_gamma_distribution
   torchtree.cli.advi.create_weibull_distribution
   torchtree.cli.advi.create_meanfield
   torchtree.cli.advi.apply_transforms_for_fullrank
   torchtree.cli.advi.create_variational_model
   torchtree.cli.advi.create_advi
   torchtree.cli.advi.create_logger
   torchtree.cli.advi.create_sampler
   torchtree.cli.advi.build_advi


Module Contents
---------------

.. py:data:: logger

.. py:function:: create_variational_parser(subprasers)

.. py:function:: create_tril(scales: torch.Tensor) -> torch.Tensor

   Create a 1 dimentional tensor containing a flatten tridiagonal matrix.

   A covariance matrix is created using scales**2 for variances and the
   covariances are set to zero. A tridiagonal is created using the
   cholesky decomposition and the diagonal elements are replaced with
   their log.

   :param scales: standard deviations
   :return:


.. py:function:: create_fullrank_from_meanfield(params, path)

.. py:function:: create_fullrank(var_id, json_object, arg)

.. py:function:: create_flexible_variational(arg, json_object)

.. py:function:: create_realnp_distribution(var_id: str, x, params: torch.Tensor)

.. py:function:: create_realnvp(var_id, json_object, arg)

.. py:function:: gather_parameters(json_object: dict, group_map: dict, parameters: dict)

.. py:function:: apply_sigmoid_transformed(json_object, value=None)

.. py:function:: apply_affine_transform(json_object, loc, scale)

.. py:function:: apply_exp_transform(json_object)

.. py:function:: apply_simplex_transform(json_object)

.. py:function:: create_normal_distribution(var_id, x_unres, json_object, loc, scale)

.. py:function:: create_gamma_distribution(var_id, x_unres, json_object, concentration, rate)

.. py:function:: create_weibull_distribution(var_id, x_unres, json_object, scale, concentration)

.. py:function:: create_meanfield(var_id: str, json_object: dict, distribution: str) -> tuple[list[str], list[str]]

.. py:function:: apply_transforms_for_fullrank(var_id: str, json_object: Union[dict, list]) -> list[tuple[str, str, list]]

.. py:function:: create_variational_model(id_, joint, arg) -> tuple[dict, list[str]]

.. py:function:: create_advi(joint, variational, parameters, arg)

.. py:function:: create_logger(id_, parameters, arg)

.. py:function:: create_sampler(id_, var_id, parameters, arg)

.. py:function:: build_advi(arg)

