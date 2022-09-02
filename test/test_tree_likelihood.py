import torch

import torchtree.evolution.tree_likelihood as likelihood
from torchtree import Parameter
from torchtree.evolution.alignment import Alignment, Sequence
from torchtree.evolution.branch_model import StrictClockModel
from torchtree.evolution.datatype import NucleotideDataType
from torchtree.evolution.io import read_tree_and_alignment
from torchtree.evolution.site_model import WeibullSiteModel
from torchtree.evolution.site_pattern import SitePattern, compress_alignment
from torchtree.evolution.substitution_model import JC69
from torchtree.evolution.taxa import Taxa, Taxon
from torchtree.evolution.tree_model import ReparameterizedTimeTreeModel, TimeTreeModel


def _prepare_tiny(tiny_newick_file, tiny_fasta_file):
    tree, dna = read_tree_and_alignment(tiny_newick_file, tiny_fasta_file, False, False)
    branch_lengths = torch.tensor(
        [
            float(node.edge_length)
            for node in sorted(
                list(tree.postorder_node_iter())[:-1], key=lambda x: x.index
            )
        ],
    )
    indices = []
    for node in tree.postorder_internal_node_iter():
        indices.append([node.index] + [child.index for child in node.child_nodes()])

    sequences = []
    taxa = []
    for taxon, seq in dna.items():
        sequences.append(Sequence(taxon.label, str(seq)))
        taxa.append(Taxon(taxon.label, None))

    partials, weights_tensor = compress_alignment(
        Alignment(None, sequences, Taxa(None, taxa), NucleotideDataType(None))
    )
    partials.extend([None] * (len(dna) - 1))
    return partials, weights_tensor, indices, branch_lengths


def test_calculate_pytorch(tiny_newick_file, tiny_fasta_file, jc69_model):
    partials, weights_tensor, indices, branch_lengths = _prepare_tiny(
        tiny_newick_file, tiny_fasta_file
    )
    mats = jc69_model.p_t(branch_lengths)
    freqs = jc69_model.frequencies.reshape(jc69_model.frequencies.shape[:-1] + (1, -1))
    log_p = likelihood.calculate_treelikelihood(
        partials, weights_tensor, indices, mats, freqs
    )
    assert torch.allclose(torch.tensor(-83.329016, dtype=log_p.dtype), log_p)


def test_calculate_pytorch_rescaled(tiny_newick_file, tiny_fasta_file, jc69_model):
    partials, weights_tensor, indices, branch_lengths = _prepare_tiny(
        tiny_newick_file, tiny_fasta_file
    )
    mats = jc69_model.p_t(branch_lengths.reshape((-1, 1)))
    freqs = jc69_model.frequencies.reshape(jc69_model.frequencies.shape[:-1] + (1, -1))
    props = torch.tensor([[[1.0]]])

    log_p = likelihood.calculate_treelikelihood_discrete(
        partials, weights_tensor, indices, mats, freqs, props
    )

    log_p_rescaled = likelihood.calculate_treelikelihood_discrete_rescaled(
        partials, weights_tensor, indices, mats, freqs, props
    )
    assert torch.allclose(log_p_rescaled, log_p)


def test_treelikelihood_json(tiny_newick_file, tiny_fasta_file):
    site_pattern = {
        'id': 'sp',
        'type': 'torchtree.evolution.site_pattern.SitePattern',
        'alignment': {
            "id": "alignment",
            "type": "torchtree.evolution.alignment.Alignment",
            'datatype': 'nucleotide',
            'file': tiny_fasta_file,
            'taxa': 'taxa',
        },
    }
    subst_model = {'id': 'm', 'type': 'torchtree.evolution.substitution_model.JC69'}
    site_model = {
        'id': 'sm',
        'type': 'torchtree.evolution.site_model.ConstantSiteModel',
    }
    tree_model = {
        'id': 'tree',
        'type': 'torchtree.evolution.tree_model.UnRootedTreeModel',
        'file': tiny_newick_file,
        'branch_lengths': {
            'id': 'branches',
            'type': 'torchtree.Parameter',
            'tensor': [0.0],
        },
        'keep_branch_lengths': True,
        'taxa': {
            'id': 'taxa',
            'type': 'torchtree.evolution.taxa.Taxa',
            'taxa': [
                {
                    "id": "A_Belgium_2_1981",
                    "type": "torchtree.evolution.taxa.Taxon",
                    "attributes": {"date": 1981},
                },
                {
                    "id": "A_ChristHospital_231_1982",
                    "type": "torchtree.evolution.taxa.Taxon",
                    "attributes": {"date": 1982},
                },
                {
                    "id": "A_Philippines_2_1982",
                    "type": "torchtree.evolution.taxa.Taxon",
                    "attributes": {"date": 1982},
                },
                {
                    "id": "A_Baylor1B_1983",
                    "type": "torchtree.evolution.taxa.Taxon",
                    "attributes": {"date": 1983},
                },
                {
                    "id": "A_Oita_3_1983",
                    "type": "torchtree.evolution.taxa.Taxon",
                    "attributes": {"date": 1983},
                },
                {
                    "id": "A_Texas_12764_1983",
                    "type": "torchtree.evolution.taxa.Taxon",
                    "attributes": {"date": 1983},
                },
                {
                    "id": "A_Alaska_8_1984",
                    "type": "torchtree.evolution.taxa.Taxon",
                    "attributes": {"date": 1984},
                },
                {
                    "id": "A_Caen_1_1984",
                    "type": "torchtree.evolution.taxa.Taxon",
                    "attributes": {"date": 1984},
                },
                {
                    "id": "A_Texas_17988_1984",
                    "type": "torchtree.evolution.taxa.Taxon",
                    "attributes": {"date": 1984},
                },
                {
                    "id": "A_Colorado_2_1987",
                    "type": "torchtree.evolution.taxa.Taxon",
                    "attributes": {"date": 1987},
                },
            ],
        },
    }
    tree_likelihood = {
        'id': 'like',
        'type': 'torchtree.tree_likelihood.TreeLikelihoodModel',
        'tree_model': tree_model,
        'site_model': site_model,
        'site_pattern': site_pattern,
        'substitution_model': subst_model,
    }

    like = likelihood.TreeLikelihoodModel.from_json(tree_likelihood, {})
    assert torch.allclose(torch.tensor([-83.329016]), like())
    like.rescale = True
    assert torch.allclose(torch.tensor([-83.329016]), like())


def test_treelikelihood_batch():
    taxa_dict = {
        'id': 'taxa',
        'type': 'torchtree.evolution.taxa.Taxa',
        'taxa': [
            {
                "id": "A",
                "type": "torchtree.evolution.taxa.Taxon",
                "attributes": {"date": 0.0},
            },
            {
                "id": "B",
                "type": "torchtree.evolution.taxa.Taxon",
                "attributes": {"date": 1.0},
            },
            {
                "id": "C",
                "type": "torchtree.evolution.taxa.Taxon",
                "attributes": {"date": 4.0},
            },
            {
                "id": "D",
                "type": "torchtree.evolution.taxa.Taxon",
                "attributes": {"date": 5.0},
            },
        ],
    }
    tree_model_dict = {
        'id': 'tree',
        'type': 'torchtree.evolution.tree_model.TimeTreeModel',
        'newick': '(((A,B),C),D);',
        'internal_heights': {
            'id': 'heights',
            'type': 'torchtree.Parameter',
            'tensor': [[10.0, 20.0, 30.0], [100.0, 200.0, 300.0]],
        },
        'taxa': taxa_dict,
    }
    tree_model_dict2 = {
        'id': 'tree',
        'type': 'torchtree.evolution.tree_model.TimeTreeModel',
        'newick': '(((A,B),C),D);',
        'internal_heights': {
            'id': 'heights',
            'type': 'torchtree.Parameter',
            'tensor': [100.0, 200.0, 300.0],
        },
        'taxa': taxa_dict,
    }
    site_pattern_dict = {
        "id": "sites",
        "type": "torchtree.evolution.site_pattern.SitePattern",
        "alignment": {
            "id": "alignment",
            "type": "torchtree.evolution.alignment.Alignment",
            'datatype': 'nucleotide',
            "taxa": 'taxa',
            "sequences": [
                {"taxon": "A", "sequence": "AAG"},
                {"taxon": "B", "sequence": "AAC"},
                {"taxon": "C", "sequence": "AAC"},
                {"taxon": "D", "sequence": "AAT"},
            ],
        },
    }
    subst_model = JC69('jc')
    # compute using a batch of 2 samples
    dic = {}
    tree_model = TimeTreeModel.from_json(tree_model_dict, dic)
    site_pattern = SitePattern.from_json(site_pattern_dict, dic)
    site_model = WeibullSiteModel(
        None, Parameter(None, torch.tensor([[1.0], [1.0]])), 4
    )
    clock_model = StrictClockModel(
        None, Parameter(None, torch.tensor([[0.01], [0.001]])), tree_model
    )
    like_batch = likelihood.TreeLikelihoodModel(
        None, site_pattern, tree_model, subst_model, site_model, clock_model
    )

    # compute using a batch of 1 sample
    # (the second sample from the previous computation)
    tree_model2 = TimeTreeModel.from_json(tree_model_dict2, {})
    site_model2 = WeibullSiteModel(None, Parameter(None, torch.tensor([1.0])), 4)
    clock_model2 = StrictClockModel(
        None, Parameter(None, torch.tensor([0.001])), tree_model2
    )
    like = likelihood.TreeLikelihoodModel(
        None, site_pattern, tree_model2, subst_model, site_model2, clock_model2
    )

    assert like() == like_batch()[1]
    like.rescale = True
    assert like() == like_batch()[1]


def test_treelikelihood_weibull(flu_a_tree_file, flu_a_fasta_file):
    taxa_list = []
    with open(flu_a_fasta_file) as fp:
        for line in fp:
            if line.startswith('>'):
                taxon = line[1:].strip()
                date = float(taxon.split('_')[-1])
                taxa_list.append(Taxon(taxon, {'date': date}))
    taxa = Taxa('taxa', taxa_list)

    site_pattern = {
        'id': 'sp',
        'type': 'torchtree.evolution.site_pattern.SitePattern',
        'alignment': {
            "id": "alignment",
            "type": "torchtree.evolution.alignment.Alignment",
            'datatype': 'nucleotide',
            'file': flu_a_fasta_file,
            'taxa': 'taxa',
        },
    }
    subst_model = JC69('jc')
    site_model = WeibullSiteModel(
        'site_model', Parameter(None, torch.tensor([[0.1]])), 4
    )
    ratios = [0.5] * 67
    root_height = [20.0]
    with open(flu_a_tree_file) as fp:
        newick = fp.read().strip()

    dic = {'taxa': taxa}
    tree_model = ReparameterizedTimeTreeModel.from_json(
        ReparameterizedTimeTreeModel.json_factory(
            'tree_model',
            newick,
            ratios,
            root_height,
            'taxa',
            **{'keep_branch_lengths': True}
        ),
        dic,
    )
    branch_model = StrictClockModel(
        None, Parameter(None, torch.tensor([[0.001]])), tree_model
    )
    dic['tree_model'] = tree_model
    dic['site_model'] = site_model
    dic['subst_model'] = subst_model
    dic['branch_model'] = branch_model

    like = likelihood.TreeLikelihoodModel.from_json(
        {
            'id': 'like',
            'type': 'torchtree.tree_likelihood.TreeLikelihoodModel',
            'tree_model': 'tree_model',
            'site_model': 'site_model',
            'site_pattern': site_pattern,
            'substitution_model': 'subst_model',
            'branch_model': 'branch_model',
        },
        dic,
    )
    assert torch.allclose(torch.tensor([-4618.2062529058]), like())
    like.rescale = True
    assert torch.allclose(torch.tensor([-4618.2062529058]), like())
    like.rescale = False

    branch_model._rates.tensor = branch_model._rates.tensor.repeat(3, 1)
    site_model._parameter.tensor = site_model._parameter.tensor.repeat(3, 1)
    tree_model._internal_heights.tensor = tree_model._internal_heights.tensor.repeat(
        3, 68
    )
    assert torch.allclose(torch.tensor([[-4618.2062529058] * 3]), like())
    like.rescale = True
    assert torch.allclose(torch.tensor([[-4618.2062529058] * 3]), like())
