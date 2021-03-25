import numpy as np
import pytest
import torch

import phylotorch.evolution.tree_likelihood as likelihood
from phylotorch.evolution.site_pattern import get_dna_leaves_partials_compressed
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
    assert -83.329016 == pytest.approx(log_p.item(), 0.0001)


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
    assert -83.329016 == pytest.approx(like().item(), 0.0001)
