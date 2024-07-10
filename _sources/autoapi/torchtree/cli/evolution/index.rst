torchtree.cli.evolution
=======================

.. py:module:: torchtree.cli.evolution


Attributes
----------

.. autoapisummary::

   torchtree.cli.evolution.logger
   torchtree.cli.evolution.COALESCENT_PIECEWISE


Functions
---------

.. autoapisummary::

   torchtree.cli.evolution.create_evolution_parser
   torchtree.cli.evolution.add_birth_death
   torchtree.cli.evolution.add_coalescent
   torchtree.cli.evolution.check_arguments
   torchtree.cli.evolution.distribution_type
   torchtree.cli.evolution.run_tree_regression
   torchtree.cli.evolution.create_tree_model
   torchtree.cli.evolution.create_poisson_tree_likelihood
   torchtree.cli.evolution.create_tree_likelihood_single
   torchtree.cli.evolution.create_tree_likelihood_general
   torchtree.cli.evolution.create_tree_likelihood
   torchtree.cli.evolution.create_site_model
   torchtree.cli.evolution.create_site_model_srd06_mus
   torchtree.cli.evolution.is_float
   torchtree.cli.evolution.create_branch_model
   torchtree.cli.evolution.build_alignment
   torchtree.cli.evolution.create_substitution_model
   torchtree.cli.evolution.create_site_pattern
   torchtree.cli.evolution.create_data_type
   torchtree.cli.evolution.create_general_data_type
   torchtree.cli.evolution.create_alignment
   torchtree.cli.evolution.create_taxa
   torchtree.cli.evolution.create_birth_death
   torchtree.cli.evolution.create_constant_birth_death
   torchtree.cli.evolution.create_bdsk
   torchtree.cli.evolution.create_coalesent
   torchtree.cli.evolution.create_substitution_model_priors
   torchtree.cli.evolution.create_ucln_prior
   torchtree.cli.evolution.parse_distribution
   torchtree.cli.evolution.create_clock_prior
   torchtree.cli.evolution.create_evolution_priors
   torchtree.cli.evolution.create_time_tree_prior
   torchtree.cli.evolution.create_bd_prior
   torchtree.cli.evolution.create_constant_bd_prior
   torchtree.cli.evolution.create_bdsk_prior
   torchtree.cli.evolution.create_poisson_evolution_joint
   torchtree.cli.evolution.create_evolution_joint
   torchtree.cli.evolution.get_engine


Module Contents
---------------

.. py:data:: logger

.. py:data:: COALESCENT_PIECEWISE
   :value: ['piecewise-constant', 'piecewise-exponential', 'piecewise-linear', 'skyglide', 'skygrid', 'skyride']


.. py:function:: create_evolution_parser(parser)

.. py:function:: add_birth_death(parser)

.. py:function:: add_coalescent(parser)

.. py:function:: check_arguments(arg, parser)

.. py:function:: distribution_type(arg, choices)

   Used by argparse for specifying distributions with optional
   parameters.


.. py:function:: run_tree_regression(arg, taxa)

.. py:function:: create_tree_model(id_: str, taxa: dict, arg)

.. py:function:: create_poisson_tree_likelihood(id_, taxa, arg)

.. py:function:: create_tree_likelihood_single(id_, tree_model, branch_model, substitution_model, site_model, site_pattern)

.. py:function:: create_tree_likelihood_general(trait: str, data_type: dict, taxa: torchtree.evolution.taxa.Taxa, arg)

.. py:function:: create_tree_likelihood(id_, taxa, alignment, arg)

.. py:function:: create_site_model(id_, arg, w=None)

.. py:function:: create_site_model_srd06_mus(id_)

.. py:function:: is_float(value)

.. py:function:: create_branch_model(id_, tree_id, taxa_count, arg, rate_init=None)

.. py:function:: build_alignment(file_name, data_type)

.. py:function:: create_substitution_model(id_, model, arg)

.. py:function:: create_site_pattern(id_, alignment, indices=None)

.. py:function:: create_data_type(id_, arg)

.. py:function:: create_general_data_type(id_, trait, taxa)

.. py:function:: create_alignment(id_, taxa, arg)

.. py:function:: create_taxa(id_, arg)

.. py:function:: create_birth_death(birth_death_id, tree_id, arg)

.. py:function:: create_constant_birth_death(birth_death_id, tree_id, arg)

.. py:function:: create_bdsk(birth_death_id, tree_id, arg)

.. py:function:: create_coalesent(id_, tree_id, taxa, arg)

.. py:function:: create_substitution_model_priors(substmodel_id, model)

.. py:function:: create_ucln_prior(branch_model_id)

.. py:function:: parse_distribution(desc)

.. py:function:: create_clock_prior(arg)

.. py:function:: create_evolution_priors(taxa, arg)

.. py:function:: create_time_tree_prior(taxa, arg)

.. py:function:: create_bd_prior(id_, parameters)

.. py:function:: create_constant_bd_prior(birth_death_id)

.. py:function:: create_bdsk_prior(birth_death_id)

.. py:function:: create_poisson_evolution_joint(taxa, arg)

.. py:function:: create_evolution_joint(taxa, alignment, arg)

.. py:function:: get_engine(arg)

   Import module or use cashed module if engine is specified in
   arguments.


