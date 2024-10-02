Tree likelihood model
=====================

**torchtree** is designed to infer parameters of phylogenetic models, with every analysis containing at least one tree likelihood object.
This object is responsible for calculating the probability of an alignment given a tree and its associated parameters.
Below, we describe the structure of a tree likelihood object through its JSON representation.

.. code-block:: JSON
    :linenos:

    [
     {
       "id": "taxa",
       "type": "Taxa",
       "taxa": [
         {
           "id": "A",
           "type": "Taxon"
         },
         {
           "id": "B",
           "type": "Taxon"
         },
         {
           "id": "C",
           "type": "Taxon"
         }
       ]
     },
     {
       "id": "alignment",
       "type": "Alignment",
       "datatype": {
         "id": "data_type",
         "type": "NucleotideDataType"
       },
       "taxa": "taxa",
       "sequences": [
         {
           "taxon": "A",
           "sequence": "ACGT"
         },
         {
           "taxon": "B",
           "sequence": "AATT"
         },
         {
           "taxon": "C",
           "sequence": "ACTT"
         }
       ]
     },
     {
       "id": "like",
       "type": "TreeLikelihoodModel",
       "tree_model": {
         "id": "tree",
         "type": "UnrootedTreeModel",
         "newick": "((A:0.1,B:0.2):0.3,C:0.4);",
         "branch_lengths": {
           "id": "branch_lengths",
           "type": "Parameter",
           "tensor": 0.1,
           "full": [
            3
          ]
         },
         "site_model": {
           "id": "sitemodel",
           "type": "ConstantSiteModel"
         },
         "substitution_model": {
           "id": "substmodel",
           "type": "GTR",
           "rates": {
             "id": "gtr_rates",
             "type": "Parameter",
             "tensor": 0.16666,
             "full": [
              6
             ]
           },
           "frequencies": {
             "id": "gtr_frequencies",
             "type": "Parameter",
             "full": 0.25,
             "tensor": [
               4
             ]
           }
         },
         "site_pattern": {
           "id": "patterns",
           "type": "SitePattern",
           "alignment": "alignment"
         }
       }
     }
    ]

The first object with type ``Taxa`` defines the taxa in the alignment. Each taxon is defined by an object with type ``Taxon`` and it might contain additional information such sampling date and geographic location.
The second object is an alignment object with type :py:class:`~torchtree.evolution.alignment.Alignment` which contains the sequences of the taxa defined in the previous object.
The third object is a tree likelihood model with type :py:class:`~torchtree.evolution.tree_likelihood.TreeLikelihoodModel` and is composed of four sub-models:

* :keycode:`tree_model`: A tree model extending the :py:class:`~torchtree.evolution.tree_model.TreeModel` class which contains the tree topology and its associated parameters.
* :keycode:`site_model`: A site model extending the :py:class:`~torchtree.evolution.site_model.SiteModel`  class which contains rate heterogeneity parameters across sites, if any.
* :keycode:`substitution_model`: A substitution model extending the :py:class:`~torchtree.evolution.substitution_model.abstract.SubstitutionModel` class which contains the paramteres that parameterize the substitution process.
* :keycode:`site_pattern`: A site pattern model extending the :py:class:`~torchtree.evolution.site_pattern.SitePattern` class which contains the compressed alignment defined in the alignment object.

An optional sub-model extending the :py:class:`~torchtree.evolution.branch_model.BranchModel` class can be added to the tree likelihood model to model the rate of evolution across branches using the :keycode:`branch_model` key.

In the JSON object above, we have specified a tree likelihood model for an unrooted tree with a GTR substitution model and equal rate across sites.

This modular design allows the definition of different tree likelihood models using different combinations of the sub-models.

For example if we wanted to define a tree likelihood model with a proportion of invariant sites we would change the value of the :keycode:`site_model` key to:

.. code-block:: JSON

    {
      "id": "sitemodel",
      "type": "InvariantSiteModel",
      "invariant": {
        "id": "proportion",
        "type": "Parameter",
        "tensor": 0.5
      }
    }