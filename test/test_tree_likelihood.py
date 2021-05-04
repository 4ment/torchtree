import numpy as np
import torch

import phylotorch.evolution.tree_likelihood as likelihood
from phylotorch import Parameter
from phylotorch.evolution.branch_model import StrictClockModel
from phylotorch.evolution.site_model import WeibullSiteModel
from phylotorch.evolution.site_pattern import get_dna_leaves_partials_compressed, SitePattern
from phylotorch.evolution.substitution_model import JC69
from phylotorch.evolution.tree_model import TimeTreeModel
from phylotorch.io import read_tree_and_alignment


def test_calculate_pytorch(tiny_newick_file, tiny_fasta_file, jc69_model):
    tree, dna = read_tree_and_alignment(tiny_newick_file, tiny_fasta_file, False, False)
    branch_lengths = np.array(
        [float(node.edge_length) for node in sorted(list(tree.postorder_node_iter())[:-1], key=lambda x: x.index)])
    indices = []
    for node in tree.postorder_internal_node_iter():
        indices.append([node.index] + [child.index for child in node.child_nodes()])
    branch_lengths_tensor = torch.tensor(branch_lengths, requires_grad=True)
    partials_tensor, weights_tensor = get_dna_leaves_partials_compressed(dna)
    mats = jc69_model.p_t(branch_lengths_tensor)
    log_p = likelihood.calculate_treelikelihood(partials_tensor, weights_tensor, indices, mats, jc69_model.frequencies)
    assert torch.allclose(torch.tensor(-83.329016, dtype=log_p.dtype), log_p)


def test_treelikelihood_json(tiny_newick_file, tiny_fasta_file):
    site_pattern = {
        'id': 'sp',
        'type': 'phylotorch.evolution.site_pattern.SitePattern',
        'datatype': 'nucleotide',
        'alignment': {
            "id": "alignment",
            "type": "phylotorch.evolution.alignment.Alignment",
            'file': tiny_fasta_file,
            'taxa': 'taxa'
        }
    }
    subst_model = {
        'id': 'm',
        'type': 'phylotorch.evolution.substitution_model.JC69'
    }
    site_model = {
        'id': 'sm',
        'type': 'phylotorch.evolution.site_model.ConstantSiteModel'
    }
    tree_model = {
        'id': 'tree',
        'type': 'phylotorch.evolution.tree_model.UnRootedTreeModel',
        'file': tiny_newick_file,
        'branch_lengths': {'id': 'branches'},
        'taxa': {
            'id': 'taxa',
            'type': 'phylotorch.evolution.taxa.Taxa',
            'taxa': [
                {"id": "A_Belgium_2_1981", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 1981}},
                {"id": "A_ChristHospital_231_1982", "type": "phylotorch.evolution.taxa.Taxon",
                 "attributes": {"date": 1982}},
                {"id": "A_Philippines_2_1982", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 1982}},
                {"id": "A_Baylor1B_1983", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 1983}},
                {"id": "A_Oita_3_1983", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 1983}},
                {"id": "A_Texas_12764_1983", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 1983}},
                {"id": "A_Alaska_8_1984", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 1984}},
                {"id": "A_Caen_1_1984", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 1984}},
                {"id": "A_Texas_17988_1984", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 1984}},
                {"id": "A_Colorado_2_1987", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 1987}}
            ]
        }
    }
    tree_likelihood = {
        'id': 'like',
        'type': 'phylotorch.tree_likelihood.TreeLikelihoodModel',
        'tree_model': tree_model,
        'site_model': site_model,
        'site_pattern': site_pattern,
        'substitution_model': subst_model
    }

    like = likelihood.TreeLikelihoodModel.from_json(tree_likelihood, {})
    assert torch.allclose(torch.tensor([-83.329016], dtype=torch.float64), like())


def test_treelikelihood_batch():
    taxa_dict = {
        'id': 'taxa',
        'type': 'phylotorch.evolution.taxa.Taxa',
        'taxa': [
            {"id": "A", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 0.0}},
            {"id": "B", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 1.0}},
            {"id": "C", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 4.0}},
            {"id": "D", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 5.0}}
        ]
    }
    tree_model_dict = {
        'id': 'tree',
        'type': 'phylotorch.evolution.tree_model.TimeTreeModel',
        'newick': '(((A,B),C),D);',
        'node_heights': {
            'id': 'heights',
            'type': 'phylotorch.Parameter',
            'tensor': [[10., 20., 30.], [100., 200., 300.]]
        },
        'taxa': taxa_dict
    }
    tree_model_dict2 = {
        'id': 'tree',
        'type': 'phylotorch.evolution.tree_model.TimeTreeModel',
        'newick': '(((A,B),C),D);',
        'node_heights': {
            'id': 'heights',
            'type': 'phylotorch.Parameter',
            'tensor': [100., 200., 300.]
        },
        'taxa': taxa_dict
    }
    site_pattern_dict = {
        "id": "sites",
        "type": "phylotorch.evolution.site_pattern.SitePattern",
        "datatype": "nucleotide",
        "alignment": {
            "id": "alignment",
            "type": "phylotorch.evolution.alignment.Alignment",
            "taxa": 'taxa',
            "sequences": [
                {"taxon": "A", "sequence": "AAG"},
                {"taxon": "B", "sequence": "AAC"},
                {"taxon": "C", "sequence": "AAC"},
                {"taxon": "D", "sequence": "AAT"},
            ]
        }
    }
    subst_model = JC69('jc')
    # compute using a batch of 2 samples
    dic = {}
    tree_model = TimeTreeModel.from_json(tree_model_dict, dic)
    site_pattern = SitePattern.from_json(site_pattern_dict, dic)
    site_model = WeibullSiteModel(None, Parameter(None, torch.tensor([[1.], [1.]])), 4)
    clock_model = StrictClockModel(None, Parameter(None, torch.tensor([[0.01], [.001]])), tree_model)
    like_batch = likelihood.TreeLikelihoodModel(None, site_pattern, tree_model, subst_model, site_model, clock_model)

    # compute using a batch of 1 sample (the second sample from the previous computation)
    tree_model2 = TimeTreeModel.from_json(tree_model_dict2, {})
    site_model2 = WeibullSiteModel(None, Parameter(None, torch.tensor([1.])), 4)
    clock_model2 = StrictClockModel(None, Parameter(None, torch.tensor([.001])), tree_model2)
    like = likelihood.TreeLikelihoodModel(None, site_pattern, tree_model2, subst_model, site_model2, clock_model2)

    assert like() == like_batch()[1]
