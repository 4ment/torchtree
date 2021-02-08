import numpy as np
import pytest
import torch

import phylotorch.beagle.treelikelihood as beagle
import phylotorch.evolution.treelikelihood as likelihood
from phylotorch.io import read_tree_and_alignment
from phylotorch.evolution.sitepattern import get_dna_leaves_partials_compressed
from phylotorch.evolution.substmodel import JC69
from phylotorch.evolution.tree import transform_ratios, heights_to_branch_lengths

try:
    import libsbn
    import libsbn.beagle_flags as beagle_flags
except ImportError:
    pass


@pytest.mark.beagle
def test_calculate_libsbn_unrooted(tiny_newick_file, tiny_fasta_file):
    inst = libsbn.unrooted_instance("unrooted")
    inst.read_newick_file(tiny_newick_file)
    inst.read_fasta_file(tiny_fasta_file)
    spec = libsbn.PhyloModelSpecification(substitution="JC69", site="constant", clock="none")
    inst.prepare_for_phylo_likelihood(spec, 1, [beagle_flags.VECTOR_SSE])

    branch_lengths_libsbn = np.array(inst.tree_collection.trees[0].branch_lengths, copy=True)
    treelike = beagle.TreelikelihoodAutogradFunction.apply
    bl_tensor = torch.tensor(branch_lengths_libsbn[:-1], requires_grad=True)
    log_p = treelike(inst, bl_tensor)
    assert -83.329016 == pytest.approx(log_p.item(), 0.0001)


@pytest.mark.beagle
def test_calculate_libsbn_unrooted_gtr(tiny_newick_file, tiny_fasta_file):
    inst = libsbn.unrooted_instance("unrooted")
    inst.read_newick_file(tiny_newick_file)
    inst.read_fasta_file(tiny_fasta_file)
    spec = libsbn.PhyloModelSpecification(substitution="GTR", site="constant", clock="none")
    inst.prepare_for_phylo_likelihood(spec, 1, [beagle_flags.VECTOR_SSE])

    branch_lengths_libsbn = np.array(inst.tree_collection.trees[0].branch_lengths, copy=True)
    treelike = beagle.TreelikelihoodAutogradFunction.apply
    rates = torch.tensor(np.array([1., 1., 1., 1., 1., 1.]))
    frequencies = torch.tensor(np.array([0.25] * 4))
    bl_tensor = torch.tensor(branch_lengths_libsbn[:-1], requires_grad=True)
    log_p = treelike(inst, bl_tensor, None, rates, frequencies)
    assert -83.329016 == pytest.approx(log_p.item(), 0.0001)


@pytest.mark.beagle
def test_calculate_libsbn_rooted(flu_a_tree_file, flu_a_fasta_file):
    inst_rooted = libsbn.rooted_instance("rooted")
    inst_rooted.read_newick_file(flu_a_tree_file)
    inst_rooted.read_fasta_file(flu_a_fasta_file)
    spec_rooted = libsbn.PhyloModelSpecification(substitution="JC69", site="constant", clock="strict")
    inst_rooted.prepare_for_phylo_likelihood(spec_rooted, 1, [beagle_flags.VECTOR_SSE])

    height_ratios_libsbn = np.array(inst_rooted.tree_collection.trees[0].height_ratios, copy=True)
    treelike = beagle.TreelikelihoodAutogradFunction.apply
    height_ratios_tensor = torch.tensor(height_ratios_libsbn, requires_grad=True)
    clock_rate_tensor = torch.tensor(np.array([0.001]), requires_grad=True)
    log_p = treelike(inst_rooted, height_ratios_tensor, clock_rate_tensor)
    # assert -4777.616349 == pytest.approx(log_p.item(), 0.0001)  # without jacobian
    assert -4786.86767578125 == pytest.approx(log_p.item(), 0.0001)


@pytest.mark.beagle
@pytest.mark.parametrize("rate", [0.00001, 0.0001])
def test_compare_libsbn_pytorch_jacobian_gradient(flu_a_tree_file, flu_a_fasta_file, rate):
    inst_rooted = libsbn.rooted_instance("rooted")
    inst_rooted.read_newick_file(flu_a_tree_file)
    inst_rooted.read_fasta_file(flu_a_fasta_file)
    spec_rooted = libsbn.PhyloModelSpecification(substitution="JC69", site="constant", clock="strict")
    inst_rooted.prepare_for_phylo_likelihood(spec_rooted, 1, [beagle_flags.VECTOR_SSE])
    inst_rates = np.array(inst_rooted.tree_collection.trees[0].rates, copy=False)
    inst_rates[:] = rate

    log_likelihood = np.array(inst_rooted.log_likelihoods())[0]

    assert log_likelihood == np.array(inst_rooted.log_likelihoods())[0]

    gradient = inst_rooted.phylo_gradients()[0]
    # jacobian_grad2 = np.array(gradient.det_jacobian_ratios_root_height)
    log_likelihood2 = np.array(gradient.log_likelihood)

    assert log_likelihood == pytest.approx(log_likelihood2, 1.0e-10)

    ratios_grad = np.array(gradient.ratios_root_height)
    clock_rate_grad = torch.tensor(np.array(gradient.clock_model))

    tree, aln = read_tree_and_alignment(flu_a_tree_file, flu_a_fasta_file)
    partials_tensor, weights_tensor = get_dna_leaves_partials_compressed(aln)

    taxa_count = len(tree.taxon_namespace)
    bounds = [None] * (2 * taxa_count - 1)
    node_heights = [None] * (2 * taxa_count - 1)

    # postorder for peeling
    post_indexing = []
    for node in tree.postorder_node_iter():
        if node.is_leaf():
            bounds[node.index] = node_heights[node.index] = node.date
        else:
            bounds[node.index] = np.max([bounds[x.index] for x in node.child_node_iter()])
            children = node.child_nodes()
            post_indexing.append((node.index, children[0].index, children[1].index))
            node_heights[node.index] = node_heights[children[0].index] + children[0].edge_length

    # preoder indexing to go from ratios to heights
    pre_indexing = []
    ratios = [None] * (taxa_count - 2)
    for node in tree.preorder_node_iter(lambda x: x != tree.seed_node):
        pre_indexing.append((node.parent_node.index, node.index))
        if not node.is_leaf():
            ratios[node.index - taxa_count] = (node_heights[node.index] - bounds[node.index]) / (
                    node_heights[node.parent_node.index] - bounds[node.index])

    pre_indexing = np.array(pre_indexing)
    root_height = torch.tensor(np.array([node_heights[-1]]), requires_grad=True)
    ratios = torch.tensor(np.array(ratios), requires_grad=True)
    clock_rate = torch.tensor(np.array([rate]), requires_grad=True)
    bounds = torch.tensor(np.array(bounds))
    node_heights = transform_ratios(torch.cat((ratios, root_height)), bounds, pre_indexing)

    indices_for_jac = [None] * (taxa_count - 2)
    for idx_parent, idx in pre_indexing:
        if idx >= taxa_count:
            indices_for_jac[idx - taxa_count] = idx_parent - taxa_count

    log_det_jacobian = torch.log(node_heights[indices_for_jac] - bounds[taxa_count:-1]).sum()

    subst_model = JC69('jc')
    branch_lengths_tensor = heights_to_branch_lengths(node_heights, bounds, pre_indexing)
    mats = subst_model.p_t(branch_lengths_tensor * clock_rate)
    log_p = likelihood.calculate_treelikelihood(partials_tensor, weights_tensor, post_indexing, mats,
                                                subst_model.frequencies) + log_det_jacobian

    assert log_likelihood == pytest.approx(log_p.detach().numpy(), 1.0e-10)
    log_p.backward()
    for a, b in zip(ratios_grad, ratios.grad.numpy()):
        assert a == pytest.approx(b, 1.0e-6)

    for a, b in zip(clock_rate_grad, clock_rate.grad.numpy()):
        assert a == pytest.approx(b, 1.0e-6)

    treelike = beagle.TreelikelihoodAutogradFunction.apply
    log_p = treelike(inst_rooted, torch.cat((ratios, root_height)), clock_rate, None, None)
    assert log_likelihood == pytest.approx(log_p.detach().numpy(), 1.0e-10)


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
        'type': 'phylotorch.evolution.sitepattern.SitePattern',
        'file': tiny_fasta_file,
        'datatype': 'nucleotide',
        'taxa': 'taxa'
    }
    subst_model = {
        'id': 'm',
        'type': 'phylotorch.evolution.substmodel.JC69'
    }
    site_model = {
        'id': 'sm',
        'type': 'phylotorch.evolution.sitemodel.ConstantSiteModel'
    }
    tree_model = {
        'id': 'tree',
        'type': 'phylotorch.evolution.tree.UnRootedTreeModel',
        'file': tiny_newick_file,
        'branch_lengths': {'id': 'branches'},
        'taxa': {
            'id': 'taxa',
            'type': 'phylotorch.evolution.taxa.Taxa',
            'taxa': [
                {"id": "A_Belgium_2_1981", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 1981}},
                {"id": "A_ChristHospital_231_1982", "type": "phylotorch.evolution.taxa.Taxon", "attributes": {"date": 1982}},
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
        'type': 'phylotorch.treelikelihood.TreeLikelihoodModel',
        'tree': tree_model,
        'sitemodel': site_model,
        'sitepattern': site_pattern,
        'substitutionmodel': subst_model
    }

    like = likelihood.TreeLikelihoodModel.from_json(tree_likelihood, {})
    assert -83.329016 == pytest.approx(like().item(), 0.0001)
