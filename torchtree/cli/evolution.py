import argparse
import csv
import importlib
import logging
import math
import numbers
import re
import sys

import numpy as np
import torch
from dendropy import TaxonNamespace, Tree

from torchtree import Parameter, ViewParameter
from torchtree.cli import PLUGIN_MANAGER
from torchtree.cli.argparse_utils import list_of_float, str_or_float, zero_or_path
from torchtree.cli.priors import create_clock_horseshoe_prior, create_one_on_x_prior
from torchtree.cli.utils import convert_date_to_real, read_dates_from_csv
from torchtree.core.utils import process_object
from torchtree.distributions import Distribution
from torchtree.distributions.ctmc_scale import CTMCScale
from torchtree.evolution.alignment import (
    Alignment,
    calculate_F3x4,
    calculate_frequencies,
    calculate_kappa,
    calculate_substitutions,
    read_fasta_sequences,
)
from torchtree.evolution.coalescent import (
    ConstantCoalescent,
    PiecewiseConstantCoalescent,
)
from torchtree.evolution.datatype import CodonDataType, NucleotideDataType
from torchtree.evolution.io import extract_taxa
from torchtree.evolution.taxa import Taxa, Taxon
from torchtree.evolution.tree_model import (
    ReparameterizedTimeTreeModel,
    UnRootedTreeModel,
    initialize_dates_from_taxa,
    setup_indexes,
)
from torchtree.evolution.tree_regression import linear_regression

logger = logging.getLogger(__name__)


_engine = None


def create_evolution_parser(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-i', '--input', required=False, help="""alignment file""")
    group.add_argument(
        '--poisson',
        action="store_true",
        help="""use poisson tree likelihood""",
    )
    parser.add_argument('-t', '--tree', required=True, help="""tree file""")
    parser.add_argument(
        '-m',
        '--model',
        choices=['JC69', 'K80', 'HKY', 'SYM', 'GTR', 'SRD06']
        + ['MG94']
        + ['LG', 'WAG'],
        default='JC69',
        help="""substitution model [default: %(default)s]""",
    )
    parser.add_argument(
        '-I',
        '--invariant',
        action='store_true',
        help="""include a proportion of invariant sites""",
    )
    parser.add_argument(
        '-f',
        '--frequencies',
        help="""frequencies""",
    )
    parser.add_argument(
        '-C',
        '--categories',
        metavar='C',
        type=int,
        default=1,
        help="""number of rate categories [default: %(default)s]""",
    )
    parser.add_argument(
        '--brlenspr',
        choices=['exponential', 'gammadir'],
        default='exponential',
        help="""prior on branch lengths of an unrooted tree [default: %(default)s]""",
    )
    parser.add_argument(
        '--brlens_init',
        type=lambda x: str_or_float(x, 'tree'),
        help="""initialize branch lengths using input tree file or the same
        value for every branch""",
    )
    parser.add_argument(
        '--clock',
        choices=['strict', 'ucln', 'horseshoe'],
        help="""type of clock""",
    )
    parser.add_argument(
        '--clockpr',
        default='ctmcscale',
        type=lambda x: distribution_type(x, ('exponential', 'ctmcscale')),
        help="""prior on substitution rate [default: %(default)s]""",
    )
    parser.add_argument(
        '--heights',
        choices=['ratio', 'shift'],
        default='ratio',
        help="""type of node height reparameterization [default: %(default)s]""",
    )
    parser.add_argument(
        '--heights_init',
        choices=['tree', 'regression'],
        help="""initialize node heights using input tree file or
         root to tip regression""",
    )
    parser.add_argument(
        '--root_height_init',
        type=float,
        help="""initialize root height""",
    )
    parser.add_argument('--rate', type=float, help="""fixed substitution rate""")
    parser.add_argument(
        '--rate_init',
        type=lambda x: str_or_float(x, 'regression'),
        help="""initialize substitution rate using 'regression' or with a value""",
    )
    parser.add_argument(
        '--dates',
        type=zero_or_path,
        help="""a csv file or 0 for contemporaneous taxa""",
    )
    parser.add_argument(
        '--date_format',
        default=None,
        help="""format of the date (yyyy/MM/dd or dd/MM/yyyy or dd-MM-yyyy)""",
    )
    parser.add_argument(
        '--date_regex',
        default=None,
        help="""regular expression to capture sampling date in sequence names""",
    )
    parser.add_argument(
        '--genetic_code',
        type=int,
        help="""index of genetic code""",
    )
    parser.add_argument(
        '--keep', action="store_true", help="""use branch length as starting values"""
    )
    parser.add_argument(
        '--use_path',
        action="store_true",
        help="""specify the alignment path instead of embedding it in the JSON file""",
    )
    parser.add_argument(
        '--use_ambiguities',
        action="store_true",
        help="""use nucleotide ambiguity codes""",
    )
    parser.add_argument(
        '--use_tip_states',
        action="store_true",
        help="""use tip states instead of tip partials""",
    )
    parser.add_argument(
        '--engine',
        help="""specify the package name of a plugin""",
    )
    parser.add_argument(
        '--include_jacobian',
        action="store_true",
        help="""include Jacobian of the node height transform""",
    )

    parser.add_argument(
        '--location_regex',
        default=None,
        help="""regular expression to capture sampling location sequence names""",
    )
    parser.add_argument(
        '--metadata',
        default=None,
        help="""csv file containing metadata (e.g. date, location)""",
    )
    parser.add_argument(
        '--trait',
        nargs='+',
        help="""column name in metadata file""",
    )
    parser = add_coalescent(parser)

    parser = add_birth_death(parser)

    parser.add_argument(
        '--grid',
        type=int,
        help="""number of grid points (number of segments) for skygrid and BDSK""",
    )
    parser.add_argument(
        '--disable_time_aware',
        action="store_true",
        help="""disable time aware skyride""",
    )
    parser.add_argument(
        '--disable_gmrf_rescaling',
        action="store_true",
        help="""disable rescaling using root height of time aware skyride""",
    )

    return parser


def add_birth_death(parser):
    parser.add_argument(
        '--birth-death',
        choices=['constant', 'bdsk'],
        default=None,
        help="""type of birth death model""",
    )
    return parser


def add_coalescent(parser):
    parser.add_argument(
        '--coalescent',
        choices=[
            "constant",
            "exponential",
            "skyride",
            "skygrid",
            "piecewise-exponential",
            "piecewise-linear",
        ],
        default=None,
        help="""type of coalescent""",
    )
    parser.add_argument(
        '--cutoff',
        type=float,
        help="""a cutoff for skygrid""",
    )
    parser.add_argument(
        '--gmrf_integrated',
        action='store_true',
        help="""use GMRF with precision integrated out""",
    )
    parser.add_argument(
        '--coalescent_non_centered',
        action='store_true',
        help="""use non-centered parameterization of population size parameters""",
    )
    parser.add_argument(
        '--coalescent_integrated',
        type=lambda x: list_of_float(x, 2),
        help="""provide the two parameters of the inverse-gamma distribution to
        integrate the population size out in constant coalescent. Values must be
        separated by a comma""",
    )
    parser.add_argument(
        '--coalescent_init',
        type=lambda x: str_or_float(x, ("tree", "constant")),
        help="""initialize coalescent parameter from input tree using MLE.
        heights_init=tree must be specified.""",
    )
    parser.add_argument(
        '--coalescent_temperature',
        type=float,
        help="""Soft coalescent""",
    )
    return parser


def distribution_type(arg, choices):
    """Used by argparse for specifying distributions with optional
    parameters."""
    res = arg.split('(')
    if (isinstance(choices, tuple) and res[0] in choices) or res[0] == choices:
        return arg
    else:
        if isinstance(choices, tuple):
            message = "'" + "','".join(choices) + '"'
        else:
            message = "'" + choices + "'"
        raise argparse.ArgumentTypeError(
            'invalid choice (choose from a number or ' + message + ')'
        )


def run_tree_regression(arg, taxa):
    taxon_namespace = TaxonNamespace([taxon['id'] for taxon in taxa['taxa']])
    tree_format = 'newick'
    with open(arg.tree) as fp:
        if next(fp).upper().startswith('#NEXUS'):
            tree_format = 'nexus'
    if tree_format == 'nexus':
        tree = Tree.get(
            path=arg.tree,
            schema='nexus',
            tree_offset=0,
            preserve_underscores=True,
            taxon_namespace=taxon_namespace,
        )
    else:
        tree = Tree.get(
            path=arg.tree,
            schema='newick',
            tree_offset=0,
            preserve_underscores=True,
            taxon_namespace=taxon_namespace,
        )
    tree.resolve_polytomies(update_bipartitions=True)
    setup_indexes(tree, False)
    taxa2 = [{'date': taxon['attributes']['date']} for taxon in taxa['taxa']]
    initialize_dates_from_taxa(tree, taxa2)
    rate, root_height = linear_regression(tree)
    return rate.item(), root_height.item()


def create_tree_model(id_: str, taxa: dict, arg):
    tree_format = 'newick'
    with open(arg.tree, 'r') as fp:
        if next(fp).upper().startswith('#NEXUS'):
            tree_format = 'nexus'
    if tree_format == 'nexus':
        tree = Tree.get(
            path=arg.tree,
            schema=tree_format,
            tree_offset=0,
            preserve_underscores=True,
        )
        newick = str(tree) + ';'
    else:
        with open(arg.tree, 'r') as fp:
            newick = fp.read()
            newick = newick.strip()

    kwargs = {}
    if arg.keep or arg.heights_init == 'tree':
        kwargs['keep_branch_lengths'] = True

    if arg.clock is not None:
        dates = [taxon['attributes']['date'] for taxon in taxa['taxa']]
        offset = max(dates) - min(dates)

        if arg.heights == 'ratio':
            ratios = Parameter.json_factory(
                f'{id_}.ratios', **{'tensor': 0.1, 'full': [len(dates) - 2]}
            )
            ratios['lower'] = 0.0
            ratios['upper'] = 1.0

            root_height = Parameter.json_factory(
                f'{id_}.root_height', **{'tensor': [offset + 1.0]}
            )
            if arg.root_height_init is not None:
                root_height['tensor'] = [arg.root_height_init]
            elif arg.cutoff is not None:
                root_height['tensor'] = [max(arg.cutoff, offset)]

            root_height['lower'] = offset
            tree_model = ReparameterizedTimeTreeModel.json_factory(
                id_, newick, 'taxa', ratios=ratios, root_height=root_height, **kwargs
            )

            if arg.heights_init == 'tree':
                # instantiate tree model with keep_branch_lengths to get the
                # ratios/root height parameters
                taxa_obj = Taxa.from_json(taxa, {})
                tree_model_obj: ReparameterizedTimeTreeModel = (
                    ReparameterizedTimeTreeModel.from_json(
                        tree_model, {'taxa': taxa_obj}
                    )
                )

                ratios = Parameter.json_factory(
                    f'{id_}.ratios',
                    **{'tensor': tree_model_obj._internal_heights.tensor[:-1].tolist()},
                )
                ratios['lower'] = 0.0
                ratios['upper'] = 1.0

                root_height = Parameter.json_factory(
                    f'{id_}.root_height',
                    **{'tensor': tree_model_obj._internal_heights.tensor[-1:].tolist()},
                )
                root_height['lower'] = offset
                tree_model = ReparameterizedTimeTreeModel.json_factory(
                    id_,
                    newick,
                    'taxa',
                    ratios=ratios,
                    root_height=root_height,
                    **kwargs,
                )
        elif arg.heights == 'shift':
            shifts = Parameter.json_factory(
                f'{id_}.shifts', **{'tensor': 0.1, 'full': [len(dates) - 1]}
            )
            shifts['lower'] = 0.0

            tree_model = ReparameterizedTimeTreeModel.json_factory(
                id_, newick, 'taxa', shifts=shifts, **kwargs
            )
            if arg.heights_init == 'tree' or arg.root_height_init is not None:
                # instantiate tree model with keep_branch_lengths to get the
                # ratios/root height parameters
                taxa_obj = Taxa.from_json(taxa, {})
                tree_model_obj: ReparameterizedTimeTreeModel = (
                    ReparameterizedTimeTreeModel.from_json(
                        tree_model, {'taxa': taxa_obj}
                    )
                )
                del tree_model["shifts"]["full"]
                if arg.heights_init == 'tree':
                    tree_model["shifts"][
                        "tensor"
                    ] = tree_model_obj._internal_heights.tensor.tolist()
                else:
                    heights = tree_model_obj.node_heights[len(taxa['taxa']) :]
                    heights /= heights[-1] / arg.root_height_init
                    tree_model["shifts"]["tensor"] = tree_model_obj.transform.inv(
                        heights
                    ).tolist()

        if arg.coalescent_init is not None and isinstance(
            arg.coalescent_init, numbers.Number
        ):
            arg._coalescent_init = arg.coalescent_init
        elif arg.coalescent_init == "tree" and arg.coalescent == "constant":
            arg._coalescent_init = ConstantCoalescent.maximum_likelihood(
                tree_model_obj.node_heights
            ).item()
        elif arg.coalescent_init == "tree" and arg.coalescent == 'skyride':
            arg._coalescent_init = PiecewiseConstantCoalescent.maximum_likelihood(
                tree_model_obj.node_heights
            )
        elif arg.coalescent_init == "constant":
            arg._coalescent_init = ConstantCoalescent.maximum_likelihood(
                tree_model_obj.node_heights
            ).item()
    else:
        brlens = 0.1
        if isinstance(arg.brlens_init, float):
            brlens = arg.brlens_init

        branch_lengths = Parameter.json_factory(
            f'{id_}.blens', **{'tensor': brlens, 'full': [len(taxa['taxa']) * 2 - 3]}
        )
        branch_lengths['lower'] = 0.0
        tree_model = UnRootedTreeModel.json_factory(
            id_, newick, branch_lengths, 'taxa', **kwargs
        )

        if arg.keep or arg.brlens_init == "tree":
            tree_model = UnRootedTreeModel.json_factory(
                id_, newick, branch_lengths, "taxa", **kwargs
            )
            taxa_obj = Taxa.from_json(taxa, {})
            tree_model_obj: UnRootedTreeModel = UnRootedTreeModel.from_json(
                tree_model, {"taxa": taxa_obj}
            )
            blens = torch.clamp(tree_model_obj._branch_lengths.tensor, min=1.0e-7)
            tree_model["branch_lengths"] = Parameter.json_factory(
                f"{id_}.blens",
                **{"tensor": blens.tolist()},
            )
            tree_model["branch_lengths"]["lower"] = 0.0

    return tree_model


def create_poisson_tree_likelihood(id_, taxa, arg):
    tree_id = 'tree'
    tree_model = create_tree_model(tree_id, taxa, arg)
    branch_model = create_branch_model('branchmodel', tree_id, len(taxa['taxa']), arg)

    treelikelihood_model = {
        'id': id_,
        'type': 'PoissonTreeLikelihood',
        'tree_model': tree_model,
        'branch_model': branch_model,
    }

    return treelikelihood_model


def create_tree_likelihood_single(
    id_, tree_model, branch_model, substitution_model, site_model, site_pattern
):
    treelikelihood_model = {
        'id': id_,
        'type': 'TreeLikelihoodModel',
        'tree_model': tree_model,
        'site_model': site_model,
        'substitution_model': substitution_model,
        'site_pattern': site_pattern,
    }
    if branch_model is not None:
        treelikelihood_model['branch_model'] = branch_model

    return treelikelihood_model


def create_tree_likelihood_general(trait: str, data_type: dict, taxa: Taxa, arg):
    site_pattern = {
        'id': f'site_pattern.{trait}',
        'type': 'AttributePattern',
        'taxa': 'taxa',
        'data_type': data_type,
        'attribute': trait,
    }
    site_model = create_site_model(f'sitemodel.{trait}', arg)

    state_count = len(data_type['codes'])
    mapping = np.arange(state_count * (state_count - 1))
    substitution_model = {
        'id': f'substmodel.{trait}',
        'type': 'GeneralNonSymmetricSubstitutionModel',
        'mapping': mapping.tolist(),
        'rates': {
            'id': f'substmodel.{trait}.rates',
            'type': 'Parameter',
            'full': [len(mapping)],
            'tensor': 1.0,
            'lower': 0.0,
        },
        'frequencies': {
            'id': f'substmodel.{trait}.freqs',
            'type': 'Parameter',
            'full': [state_count],
            'tensor': 1.0 / state_count,
            'lower': 0.0,
            'upper': 0.0
            # 'simplex': True
        },
        'state_count': state_count,
    }
    # substitution_model = {
    #     'id': f'substmodel.{trait}',
    #     'type': 'GeneralJC69',
    #     'state_count': len(data_type['codes']),
    # }

    treelikelihood_model = {
        'id': f'like.{trait}',
        'type': 'TreeLikelihoodModel',
        'tree_model': 'tree',
        'site_model': site_model,
        'substitution_model': substitution_model,
        'site_pattern': site_pattern,
    }
    if arg.clock is not None:
        treelikelihood_model['branch_model'] = 'branchmodel'

    if arg.use_ambiguities:
        treelikelihood_model['use_ambiguities'] = True
    if arg.use_tip_states:
        treelikelihood_model['use_tip_states'] = True

    return treelikelihood_model


def create_tree_likelihood(id_, taxa, alignment, arg):
    rate_init = None
    if arg.clock is not None and (
        arg.rate_init == 'regression' or arg.heights_init == 'regression'
    ):
        dates = [taxon['attributes']['date'] for taxon in taxa['taxa']]
        # only use regression for heterochronous data
        if max(dates) != min(dates):
            rate_init_r, root_height_init = run_tree_regression(arg, taxa)
            if arg.rate_init is None:
                rate_init = rate_init_r
            if arg.root_height_init is None:
                arg.root_height_init = max(dates) - root_height_init
    else:
        rate_init = arg.rate_init

    if arg.model == 'SRD06':
        branch_model = None
        branch_model_id = None
        tree_id = 'tree'
        tree_model = create_tree_model(tree_id, taxa, arg)
        if arg.clock is not None:
            branch_model_id = 'branchmodel'
            branch_model = create_branch_model(
                branch_model_id, tree_id, len(taxa['taxa']), arg, rate_init
            )

        like_list = []
        for tag, indices, t, b, w in zip(
            ('12', '3'),
            ('::3,1::3', '2::3'),
            (tree_model, tree_id),
            (branch_model, branch_model_id),
            ('0:1', '1:2'),
        ):
            substitution_model = create_substitution_model(
                f'substmodel.{tag}', 'HKY', arg
            )
            site_model = create_site_model(
                f'sitemodel.{tag}',
                arg,
                ViewParameter.json_factory(f'sitemodel.{tag}.mu', 'srd06.mus', w),
            )
            site_pattern = create_site_pattern(f'patterns.{tag}', alignment, indices)
            like_list.append(
                create_tree_likelihood_single(
                    f'{id_}.{tag}', t, b, substitution_model, site_model, site_pattern
                )
            )
            for plugin in PLUGIN_MANAGER.plugins():
                plugin.process_tree_likelihood(arg, like_list[-1])

        joint_like = {
            'id': 'like',
            'type': 'JointDistributionModel',
            'distributions': like_list,
        }
        return joint_like

    site_pattern = create_site_pattern('patterns', alignment)
    site_model = create_site_model('sitemodel', arg)
    substitution_model = create_substitution_model('substmodel', arg.model, arg)
    tree_id = 'tree'
    tree_model = create_tree_model(tree_id, taxa, arg)

    treelikelihood_model = {
        'id': id_,
        'type': 'TreeLikelihoodModel',
        'tree_model': tree_model,
        'site_model': site_model,
        'substitution_model': substitution_model,
        'site_pattern': site_pattern,
    }
    if arg.use_ambiguities:
        treelikelihood_model['use_ambiguities'] = True
    if arg.use_tip_states:
        treelikelihood_model['use_tip_states'] = True

    if arg.clock is not None:
        treelikelihood_model['branch_model'] = create_branch_model(
            'branchmodel', tree_id, len(taxa['taxa']), arg, rate_init
        )

    for plugin in PLUGIN_MANAGER.plugins():
        plugin.process_tree_likelihood(arg, treelikelihood_model)

    evol = get_engine(arg)
    if evol is not None:
        try:
            evol.process_tree_likelihood(arg, treelikelihood_model)
        except AttributeError as e:
            sys.stderr.write(str(e) + '\n')
            sys.stderr.write(
                'The process_tree_likelihood method is not implemented'
                f' in the {arg.engine} engine\n'
            )
            sys.exit(1)

    return treelikelihood_model


def create_site_model(id_, arg, w=None):
    if arg.categories == 1:
        if arg.invariant:
            prop = Parameter.json_factory(f'{id_}.pinv', **{'tensor': [0.1]})
            prop['lower'] = 0
            prop['upper'] = 1
            site_model = {'id': id_, 'type': 'InvariantSiteModel', 'invariant': prop}
        else:
            site_model = {'id': id_, 'type': 'ConstantSiteModel'}
    else:
        shape = Parameter.json_factory(f'{id_}.shape', **{'tensor': [0.1]})
        shape['lower'] = 0.0
        site_model = {
            'id': id_,
            'type': 'WeibullSiteModel',
            'categories': arg.categories,
            'shape': shape,
        }
        if arg.invariant:
            prop = Parameter.json_factory(f'{id_}.pinv', **{'tensor': [0.1]})
            prop['lower'] = 0
            prop['upper'] = 1
            site_model['invariant'] = prop

    if arg.model == 'SRD06':
        site_model['mu'] = w
    return site_model


def create_site_model_srd06_mus(id_):
    weights = [2 / 3, 1 / 3]
    y = Parameter.json_factory('srd06.mu', **{'tensor': [0.5, 0.5]})
    y['simplex'] = True
    mus = {
        'id': id_,
        'type': 'TransformedParameter',
        'transform': 'ConvexCombinationTransform',
        'x': y,
        'parameters': {'weights': weights},
    }
    return mus


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
    except TypeError:
        return False


def create_branch_model(id_, tree_id, taxa_count, arg, rate_init=None):
    if arg.rate is not None:
        rate = [arg.rate]
    elif rate_init is not None:
        rate = [rate_init]
    else:
        rate = [0.001]
    rate_parameter = Parameter.json_factory(f'{id_}.rate', **{'tensor': rate})
    rate_parameter['lower'] = 0.0

    if arg.rate is not None:
        rate_parameter['lower'] = rate_parameter['upper'] = arg.rate

    if arg.clock == 'strict':
        return {
            'id': id_,
            'type': 'StrictClockModel',
            'tree_model': tree_id,
            'rate': rate_parameter,
        }
    elif arg.clock == 'horseshoe':
        rates = Parameter.json_factory(
            f'{id_}.rates.unscaled', **{'tensor': 1.0, 'full': [2 * taxa_count - 2]}
        )
        rates['lower'] = 0.0
        rescaled_rates = {
            'id': f'{id_}.rates',
            'type': 'TransformedParameter',
            'transform': 'RescaledRateTransform',
            'x': rates,
            'parameters': {
                'tree_model': tree_id,
                'rate': rate_parameter,
            },
        }
        return {
            'id': f'{id_}.simple',
            'type': 'SimpleClockModel',
            'tree_model': tree_id,
            'rate': rescaled_rates,
        }
    elif arg.clock == 'ucln':
        rate = Parameter.json_factory(
            f'{id_}.rates', **{'tensor': 0.001, 'full': [2 * taxa_count - 2]}
        )
        rate['lower'] = 0.0
        return {
            'id': id_,
            'type': 'SimpleClockModel',
            'tree_model': tree_id,
            'rate': rate,
        }


def build_alignment(file_name, data_type):
    sequences = read_fasta_sequences(file_name)
    taxa = Taxa('taxa', [Taxon(sequence.taxon, {}) for sequence in sequences])
    return Alignment('alignment', sequences, taxa, data_type)


def create_substitution_model(id_, model, arg):
    if model == 'JC69':
        return {'id': id_, 'type': 'JC69'}
    elif model in ('K80', 'HKY', 'SYM', 'GTR'):
        frequencies = Parameter.json_factory(
            f'{id_}.frequencies', **{'tensor': [0.25] * 4}
        )
        frequencies['simplex'] = True
        alignment = None

        if arg.frequencies is not None:
            if arg.frequencies == 'empirical':
                alignment = build_alignment(arg.input, NucleotideDataType(''))
                frequencies['tensor'] = calculate_frequencies(alignment)
            elif arg.frequencies != 'equal':
                frequencies['tensor'] = list(map(float, arg.frequencies.split(',')))
                if len(frequencies['tensor']) != 4:
                    raise ValueError(
                        f'The dimension of the frequencies parameter '
                        f'({len(frequencies["tensor"])}) does not match the data type '
                        f'state count 4'
                    )

        if model in ('K80', 'HKY'):
            kappa = Parameter.json_factory(f'{id_}.kappa', **{'tensor': [3.0]})
            kappa['lower'] = 0.0
            if alignment is not None:
                kappa['tensor'] = [calculate_kappa(alignment, frequencies['tensor'])]
            if model == 'SYM':
                frequencies['lower'] = frequencies['upper'] = 1
            return {
                'id': id_,
                'type': 'HKY',
                'kappa': kappa,
                'frequencies': frequencies,
            }
        elif model in ('SYM', 'GTR'):
            rates = Parameter.json_factory(
                f'{id_}.rates', **{'tensor': 1 / 6, 'full': [6]}
            )
            rates['simplex'] = True
            mapping = ((6, 0, 1, 2), (0, 6, 3, 4), (1, 3, 6, 5), (2, 4, 5, 6))
            if alignment is not None:
                rel_rates = np.array(calculate_substitutions(alignment, mapping))
                rates['tensor'] = (rel_rates[:-1] / rel_rates[:-1].sum()).tolist()
            if model == 'SYM':
                frequencies['lower'] = frequencies['upper'] = 1
            return {
                'id': id_,
                'type': 'GTR',
                'rates': rates,
                'frequencies': frequencies,
            }
    elif model == 'MG94':
        alpha = Parameter.json_factory(f'{id_}.alpha', **{'tensor': [1.0]})
        alpha['lower'] = 0.0
        beta = Parameter.json_factory(f'{id_}.beta', **{'tensor': [1.0]})
        beta['lower'] = 0.0
        kappa = Parameter.json_factory(f'{id_}.kappa', **{'tensor': [1.0]})
        kappa['lower'] = 0.0

        data_type_json = 'data_type'
        if not hasattr(arg, '_data_type'):
            arg._data_type = create_data_type('data_type', arg)
            data_type_json = arg._data_type
        data_type = process_object(arg._data_type, {})
        frequencies = Parameter.json_factory(
            f'{id_}.frequencies',
            **{'tensor': 1 / data_type.state_count, 'full': [data_type.state_count]},
        )
        # it is a simplex but it is fixed
        frequencies['lower'] = frequencies['upper'] = 1

        if arg.frequencies is not None and arg.frequencies != 'equal':
            if arg.frequencies == 'F3x4':
                alignment = build_alignment(arg.input, data_type)
                frequencies['tensor'] = calculate_F3x4(alignment)
            else:
                frequencies['tensor'] = list(map(float, arg.frequencies.split(',')))
                if len(frequencies['tensor']) != data_type.state_count:
                    raise ValueError(
                        f'The dimension of the frequencies parameter '
                        f'({len(frequencies["tensor"])}) does not match the data type '
                        f'state count {data_type.state_count}'
                    )
            del frequencies['full']

        return {
            'id': id_,
            'type': 'MG94',
            'alpha': alpha,
            'beta': beta,
            'kappa': kappa,
            'frequencies': frequencies,
            'data_type': data_type_json,
        }
    elif model in ('LG', 'WAG'):
        return {'id': id_, 'type': model}


def create_site_pattern(id_, alignment, indices=None):
    site_pattern = {'id': id_, 'type': 'SitePattern', 'alignment': alignment}
    if indices is not None:
        site_pattern['indices'] = indices
    return site_pattern


def create_data_type(id_, arg):
    if arg.model == 'MG94':
        data_type = {
            'id': id_,
            'type': 'CodonDataType',
            'genetic_code': CodonDataType.GENETIC_CODE_NAMES[arg.genetic_code],
        }
    elif arg.model in ('LG', 'WAG'):
        data_type = {'id': id_, 'type': 'AminoAcidDataType'}
    else:
        data_type = {'id': id_, 'type': 'NucleotideDataType'}
    return data_type


def create_general_data_type(id_, trait, taxa):
    unique_codes = list(
        {
            taxon['attributes'][trait]
            for taxon in taxa['taxa']
            if taxon['attributes'][trait] not in ('', '?', '-')
        }
    )

    data_type = {
        'id': id_,
        'type': 'GeneralDataType',
        'codes': unique_codes,
    }
    return data_type


def create_alignment(id_, taxa, arg):
    data_type_json = 'data_type'
    if not hasattr(arg, '_data_type'):
        arg._data_type = create_data_type('data_type', arg)
        data_type_json = arg._data_type
    alignment = {
        'id': id_,
        'type': 'Alignment',
        'datatype': data_type_json,
        'taxa': taxa,
    }

    if arg.use_path:
        alignment['file'] = arg.input
    else:
        sequences = read_fasta_sequences(arg.input)
        sequence_list = []
        for sequence in sequences:
            sequence_list.append(
                {'taxon': sequence.taxon, 'sequence': sequence.sequence}
            )
        alignment['sequences'] = sequence_list

    return alignment


def create_taxa(id_, arg):
    if arg.input is not None:
        alignment = read_fasta_sequences(arg.input)
        taxa_list = []
        for sequence in alignment:
            taxa_list.append({'id': sequence.taxon, 'type': 'Taxon'})
    else:
        taxa = extract_taxa(arg.tree)
        taxa_list = [{'id': taxon, 'type': 'Taxon'} for taxon in taxa]

    taxa = {'id': id_, 'type': 'Taxa', 'taxa': taxa_list}
    if arg.clock is not None:
        if arg.dates == 0:
            for taxon in taxa_list:
                taxon['attributes'] = {'date': 0.0}
        elif arg.dates is not None:
            dates = read_dates_from_csv(arg.dates, arg.date_format)
            for taxon in taxa_list:
                taxon['attributes'] = {'date': dates[taxon['id']]}
        else:
            regex_date = r'_(\d+\.?\d*)$'
            if arg.date_regex is not None:
                regex_date = arg.date_regex
            regex = re.compile(regex_date)

            if arg.date_format is not None:
                res = re.split(r"[/-]", arg.date_format)
                yy = res.index('yyyy') + 1
                MM = res.index('MM') + 1
                dd = res.index('dd') + 1

            for taxon in taxa_list:
                res = re.search(regex, taxon['id'])
                if res is None:
                    logger.error(
                        f" Could not extract date from {taxon['id']}"
                        f" - regular expression used: {regex_date}"
                    )
                    sys.exit(1)
                if len(res.groups()) > 1:
                    taxon['attributes'] = {
                        'date': convert_date_to_real(
                            int(res[dd]), int(res[MM]), int(res[yy])
                        )
                    }
                else:
                    taxon['attributes'] = {'date': float(res.group(1))}

    if arg.location_regex:
        regex = re.compile(arg.location_regex)
        for taxon in taxa_list:
            res = re.search(regex, taxon['id'])
            if res is None:
                logger.error(
                    f" Could not extract location from {taxon['id']}"
                    f" - regular expression used: {arg.location_regex}"
                )
                sys.exit(1)

            if 'attributes' not in taxon:
                taxon['attributes'] = {}
            taxon['attributes']['location'] = res.group(1)
    elif arg.trait and arg.metadata:
        with open(arg.metadata) as fp:
            reader = csv.reader(
                fp,
                quotechar='"',
                delimiter=',',
                quoting=csv.QUOTE_ALL,
                skipinitialspace=True,
            )

            for line in reader:
                indices = [line.index(trait) for trait in arg.trait]
                break
            taxa_map = {taxon['id']: taxon for taxon in taxa['taxa']}
            for line in reader:
                if 'attributes' not in taxa_map[line[1]]:
                    taxa_map[line[1]]['attributes'] = {}
                for trait, idx in zip(arg.trait, indices):
                    taxa_map[line[1]]['attributes'][trait] = line[idx]
    return taxa


def create_birth_death(birth_death_id, tree_id, arg):
    if arg.birth_death == 'constant':
        return create_constant_birth_death(birth_death_id, tree_id, arg)
    elif arg.birth_death == 'bdsk':
        return create_bdsk(birth_death_id, tree_id, arg)


def create_constant_birth_death(birth_death_id, tree_id, arg):
    lambda_ = Parameter.json_factory(
        f'{birth_death_id}.lambda',
        **{'tensor': [3.0]},
    )
    lambda_['lower'] = 0.0
    mu = Parameter.json_factory(
        f'{birth_death_id}.mu',
        **{'tensor': [2.0]},
    )
    mu['lower'] = 0.0
    psi = Parameter.json_factory(
        f'{birth_death_id}.psi',
        **{'tensor': [1.0]},
    )
    psi['lower'] = 0.0
    rho = Parameter.json_factory(
        f'{birth_death_id}.rho',
        **{
            'tensor': [1.0e-6],
        },
    )
    rho['lower'] = 0.0
    rho['upper'] = 1.0

    origin = {
        'id': f'{birth_death_id}.origin',
        'type': 'TransformedParameter',
        'transform': 'torch.distributions.AffineTransform',
        'x': {
            'id': f'{birth_death_id}.origin.unshifted',
            'type': 'Parameter',
            'tensor': [1.0],
            'lower': 0.0,
        },
        'parameters': {
            'loc': f'{tree_id}.root_height',
            'scale': 1.0,
        },
    }

    bd = {
        'id': birth_death_id,
        'type': 'BirthDeathModel',
        'tree_model': tree_id,
        'lambda': lambda_,
        'mu': mu,
        'psi': psi,
        'rho': rho,
        'origin': origin,
    }
    return bd


def create_bdsk(birth_death_id, tree_id, arg):
    R = Parameter.json_factory(
        f'{birth_death_id}.R',
        **{'tensor': 3.0, 'full': [arg.grid]},
    )
    R['lower'] = 0.0
    delta = Parameter.json_factory(
        f'{birth_death_id}.delta',
        **{'tensor': 3.0, 'full': [arg.grid]},
    )
    delta['lower'] = 0.0
    s = Parameter.json_factory(
        f'{birth_death_id}.s',
        **{'tensor': 0.0, 'full': [arg.grid]},
    )
    s['lower'] = 0.0
    s['upper'] = 1.0
    # fixed to 0 for contemporaneous data
    if arg.dates == 0:
        s['upper'] = 0.0

    rho = Parameter.json_factory(
        f'{birth_death_id}.rho',
        **{
            'tensor': [1.0e-6],
        },
    )
    rho['lower'] = 0.0
    rho['upper'] = 1.0

    origin = {
        'id': f'{birth_death_id}.origin',
        'type': 'TransformedParameter',
        'transform': 'torch.distributions.AffineTransform',
        'x': {
            'id': f'{birth_death_id}.origin.unshifted',
            'type': 'Parameter',
            'tensor': [1.0],
            'lower': 0.0,
        },
        'parameters': {
            'loc': f'{tree_id}.root_height',
            'scale': 1.0,
        },
    }
    bdsk = {
        'id': birth_death_id,
        'type': 'BDSKModel',
        'tree_model': tree_id,
        'R': R,
        'delta': delta,
        's': s,
        'rho': rho,
        'origin': origin,
    }
    return bdsk


def create_coalesent(id_, tree_id, taxa, arg):
    joint_list = []
    params = {}
    if arg.coalescent in ('constant', 'exponential'):
        theta_value = arg._coalescent_init if '_coalescent_init' in arg else 100.0

        theta = Parameter.json_factory(f'{id_}.theta', **{'tensor': [theta_value]})
        theta['lower'] = 0.0
        if arg.coalescent != "constant" or arg.coalescent_integrated is None:
            joint_list.append(
                create_one_on_x_prior(f'{id_}.theta.prior', f'{id_}.theta')
            )
    if arg.coalescent == 'exponential':
        growth = Parameter.json_factory(f'{id_}.growth', **{'tensor': [0.01]})
        params['growth'] = growth

        joint_list.append(
            Distribution.json_factory(
                f'{id_}.growth.prior',
                'torch.distributions.Laplace',
                f'{id_}.growth',
                {
                    'loc': 0.0,
                    'scale': 1.0,
                },
            )
        )
    elif arg.coalescent in (
        "skyride",
        "skygrid",
        "piecewise-exponential",
        "piecewise-linear",
    ):
        if arg.coalescent == "skyride":
            theta_shape = [len(taxa["taxa"]) - 1]
        else:
            theta_shape = [arg.grid]

        if arg.coalescent_non_centered:
            theta = {
                "id": f'{id_}.theta',
                "type": "TransformedParameter",
                "transform": "CumSumExpTransform",
                "x": [
                    {"id": "theta1.unres", "type": "Parameter", "tensor": [1.0]},
                    {
                        "id": "theta.unres",
                        "type": "Parameter",
                        "tensor": 0.0,
                        "full": [theta_shape[0] - 1],
                    },
                ],
            }
        else:
            if "_coalescent_init" in arg:
                if isinstance(arg._coalescent_init, numbers.Number):
                    theta_log = Parameter.json_factory(
                        f"{id_}.theta.log",
                        **{
                            "tensor": math.log(arg._coalescent_init),
                            "full": theta_shape,
                        },
                    )
                elif isinstance(arg._coalescent_init, torch.Tensor):
                    theta_log = Parameter.json_factory(
                        f"{id_}.theta.log",
                        **{
                            "tensor": torch.clamp(arg._coalescent_init, min=1.0e-6)
                            .log()
                            .tolist()
                        },
                    )
                else:
                    sys.stderr.write("_coalescent_init is not valid\n")
                    exit(1)
            else:
                theta_log = Parameter.json_factory(
                    f"{id_}.theta.log", **{"tensor": 3.0, "full": theta_shape}
                )

            theta = {
                'id': f'{id_}.theta',
                'type': 'TransformedParameter',
                'transform': 'torch.distributions.ExpTransform',
                'x': theta_log,
            }

        if arg.gmrf_integrated:
            gmrf = {
                'id': 'gmrf',
                'type': 'GMRFGammaIntegrated',
                'x': f'{id_}.theta.log',
                'shape': 0.001,
                'rate': 0.001,
            }
        else:
            gmrf = {
                'id': 'gmrf',
                'type': 'GMRF',
                'x': f'{id_}.theta.log',
                'precision': Parameter.json_factory(
                    'gmrf.precision',
                    **{'tensor': [0.1]},
                ),
            }

        if arg.coalescent_non_centered:
            gmrf["x"] = {
                "id": f'{id_}.theta.log',
                "type": "TransformedParameter",
                "transform": "LogTransform",
                "x": f'{id_}.theta',
            }

        joint_list.append(gmrf)

        if arg.coalescent == "skyride":
            if not arg.disable_time_aware:
                gmrf['tree_model'] = 'tree'

            if arg.disable_gmrf_rescaling:
                gmrf["rescale"] = False

        if not arg.gmrf_integrated:
            gmrf['precision']['lower'] = 0.0
            joint_list.append(
                Distribution.json_factory(
                    'gmrf.precision.prior',
                    'torch.distributions.Gamma',
                    'gmrf.precision',
                    {
                        'concentration': 0.0010,
                        'rate': 0.0010,
                    },
                )
            )

    if arg.coalescent == 'constant':
        if arg.coalescent_integrated is not None:
            # alhpa 3 beta 0.003
            alpha, beta = arg.coalescent_integrated.split(",")
            coalescent = {
                "id": id_,
                "type": 'ConstantCoalescentIntegratedModel',
                "tree_model": tree_id,
                "alpha": float(alpha),
                "beta": float(beta),
            }
        else:
            coalescent = {
                'id': id_,
                'type': 'ConstantCoalescentModel',
                'theta': theta,
                'tree_model': tree_id,
            }
    elif arg.coalescent == 'exponential':
        coalescent = {
            'id': id_,
            'type': 'ExponentialCoalescentModel',
            'theta': theta,
            'growth': growth,
            'tree_model': tree_id,
        }
    elif arg.coalescent == 'skygrid':
        coalescent = {
            'id': id_,
            'type': 'PiecewiseConstantCoalescentGridModel',
            'theta': theta,
            'tree_model': tree_id,
            'cutoff': arg.cutoff,
        }
        if arg.coalescent_temperature is not None:
            coalescent['temperature'] = arg.coalescent_temperature
    elif arg.coalescent == 'skyride':
        coalescent = {
            'id': id_,
            'type': 'PiecewiseConstantCoalescentModel',
            'theta': theta,
            'tree_model': tree_id,
        }
    elif arg.coalescent == 'piecewise-exponential':
        coalescent = {
            'id': id_,
            'type': 'PiecewiseExponentialCoalescentGridModel',
            'theta': theta,
            'growth': Parameter.json_factory(
                f'{id_}.growth', **{'tensor': 0.001, 'full': [arg.grid]}
            ),
            'tree_model': tree_id,
            'cutoff': arg.cutoff,
        }
    elif arg.coalescent == 'piecewise-linear':
        coalescent = {
            'id': id_,
            'type': 'PiecewiseLinearCoalescentGridModel',
            'theta': theta,
            'tree_model': tree_id,
            'cutoff': arg.cutoff,
        }

    for plugin in PLUGIN_MANAGER.plugins():
        plugin.process_coalescent(arg, coalescent)

    evol = get_engine(arg)
    if evol is not None:
        try:
            evol.process_coalescent(arg, coalescent)
        except AttributeError:
            pass

    return [coalescent] + joint_list


def create_substitution_model_priors(substmodel_id, model):
    joint_list = []
    if model in ('HKY', 'SYM', 'GTR'):
        if model != 'SYM':
            joint_list.append(
                Distribution.json_factory(
                    f'{substmodel_id}.frequencies.prior',
                    'torch.distributions.Dirichlet',
                    f'{substmodel_id}.frequencies',
                    {'concentration': [1.0] * 4},
                )
            )

        if model in ('SYM', 'GTR'):
            joint_list.append(
                Distribution.json_factory(
                    f'{substmodel_id}.rates.prior',
                    'torch.distributions.Dirichlet',
                    f'{substmodel_id}.rates',
                    {'concentration': [1.0] * 6},
                )
            )
        else:
            joint_list.append(
                Distribution.json_factory(
                    f'{substmodel_id}.kappa.prior',
                    'torch.distributions.LogNormal',
                    f'{substmodel_id}.kappa',
                    {'loc': 1.0, 'scale': 1.25},
                )
            )
    if model == 'MG94':
        joint_list.extend(
            (
                Distribution.json_factory(
                    f'{substmodel_id}.kappa.prior',
                    'torch.distributions.LogNormal',
                    f'{substmodel_id}.kappa',
                    {'loc': 1.0, 'scale': 1.25},
                ),
                Distribution.json_factory(
                    f'{substmodel_id}.alpha.prior',
                    'torch.distributions.Gamma',
                    f'{substmodel_id}.alpha',
                    {'concentration': 0.001, 'rate': 0.001},
                ),
                Distribution.json_factory(
                    f'{substmodel_id}.beta.prior',
                    'torch.distributions.Gamma',
                    f'{substmodel_id}.beta',
                    {'concentration': 0.001, 'rate': 0.001},
                ),
            )
        )
    return joint_list


def create_ucln_prior(branch_model_id):
    joint_list = []
    mean = Parameter.json_factory(
        f'{branch_model_id}.rates.prior.mean', **{'tensor': [0.001]}
    )
    scale = Parameter.json_factory(
        f'{branch_model_id}.rates.prior.scale', **{'tensor': [1.0]}
    )
    mean['lower'] = 0.0
    scale['lower'] = 0.0
    joint_list.append(
        Distribution.json_factory(
            f'{branch_model_id}.rates.prior',
            'LogNormal',
            f'{branch_model_id}.rates',
            {
                'mean': mean,
                'scale': scale,
            },
        )
    )
    joint_list.append(
        CTMCScale.json_factory(
            f'{branch_model_id}.mean.prior',
            f'{branch_model_id}.rates.prior.mean',
            'tree',
        )
    )
    joint_list.append(
        Distribution.json_factory(
            f'{branch_model_id}.rates.scale.prior',
            'torch.distributions.Gamma',
            f'{branch_model_id}.rates.prior.scale',
            {
                'concentration': 0.5396,
                'rate': 2.6184,
            },
        )
    )
    return joint_list


def parse_distribution(desc):
    res = desc.split('(')
    if len(res) == 1:
        return res, None
    else:
        return res[0], list(map(float, res[1][:-1].split(',')))


def create_clock_prior(arg):
    branch_model_id = 'branchmodel'
    tree_id = 'tree'
    prior_list = []
    if arg.clock == 'strict' and arg.rate is None:
        if arg.clockpr == 'ctmcscale':
            prior_list.append(
                CTMCScale.json_factory(
                    f'{branch_model_id}.rate.prior', f'{branch_model_id}.rate', tree_id
                )
            )
        elif arg.clockpr.startswith('exponential'):
            name, params = parse_distribution(arg.clockpr)
            if params is None:
                rate = 1000.0
            else:
                rate = params[0]
            prior_list.append(
                Distribution.json_factory(
                    f'{branch_model_id}.rate.prior',
                    'torch.distributions.Exponential',
                    f'{branch_model_id}.rate',
                    {
                        'rate': rate,
                    },
                ),
            )

    elif arg.clock == 'ucln':
        prior_list.extend(create_ucln_prior(branch_model_id))
    elif arg.clock == 'horseshoe':
        prior_list.extend(create_clock_horseshoe_prior(branch_model_id, tree_id))
    return prior_list


def create_evolution_priors(taxa, arg):
    tree_id = 'tree'
    joint_list = []

    if arg.coalescent is not None or arg.birth_death is not None:
        joint_list.extend(create_time_tree_prior(taxa, arg))

    if arg.clock is not None:
        joint_list.extend(create_clock_prior(arg))
    else:
        if arg.brlenspr == 'exponential':
            joint_list.append(
                Distribution.json_factory(
                    f'{tree_id}.blens.prior',
                    'torch.distributions.Exponential',
                    f'{tree_id}.blens',
                    {
                        'rate': 10.0,
                    },
                ),
            )
        elif arg.brlenspr == 'gammadir':
            joint_list.append(
                {
                    'id': f'{tree_id}.blens.prior',
                    'type': 'CompoundGammaDirichletPrior',
                    'tree_model': tree_id,
                    'alpha': 1.0,
                    'c': 0.1,
                    'shape': 1.0,
                    'rate': 1.0,
                }
            )

    if arg.model == 'SRD06':
        for tag in ('12', '3'):
            joint_list.extend(
                create_substitution_model_priors(f'substmodel.{tag}', 'HKY')
            )
    else:
        joint_list.extend(create_substitution_model_priors('substmodel', arg.model))

    if arg.categories > 1:
        sitemodel_id = 'sitemodel'
        if arg.model == 'SRD06':
            for tag in ('12', '3'):
                joint_list.append(
                    Distribution.json_factory(
                        f'{sitemodel_id}.{tag}.shape.prior',
                        'torch.distributions.Exponential',
                        f'{sitemodel_id}.{tag}.shape',
                        {'rate': 2.0},
                    )
                )
        else:
            joint_list.append(
                Distribution.json_factory(
                    f'{sitemodel_id}.shape.prior',
                    'torch.distributions.Exponential',
                    f'{sitemodel_id}.shape',
                    {'rate': 2.0},
                )
            )
    return joint_list


def create_time_tree_prior(taxa, arg):
    if arg.coalescent is not None:
        joint_list = create_coalesent('coalescent', 'tree', taxa, arg)
    elif arg.birth_death is not None:
        if arg.birth_death == 'constant':
            joint_list = create_constant_bd_prior(arg.birth_death)
        elif arg.birth_death == 'bdsk':
            joint_list = create_bdsk_prior(arg.birth_death)

        prior = create_birth_death(arg.birth_death, 'tree', arg)
        joint_list.insert(0, prior)

    return joint_list


def create_bd_prior(id_, parameters):
    joint_list = []
    for name, x in parameters:
        joint_list.append(
            Distribution.json_factory(
                f'{id_}.{name}.prior',
                'torch.distributions.LogNormal',
                f'{id_}.{x}',
                {
                    'loc': 1.0,
                    'scale': 1.25,
                },
            ),
        )
    joint_list.append(
        Distribution.json_factory(
            f'{id_}.rho.prior',
            'torch.distributions.Beta',
            f'{id_}.rho',
            {
                'concentration1': 1.0,
                'concentration0': 9999.0,
            },
        ),
    )
    return joint_list


def create_constant_bd_prior(birth_death_id):
    return create_bd_prior(
        birth_death_id,
        (('lambda', 'lambda'), ('mu', 'mu'), ('origin', 'origin.unshifted')),
    )


def create_bdsk_prior(birth_death_id):
    return create_bd_prior(
        birth_death_id, (('R', 'R'), ('delta', 'delta'), ('origin', 'origin.unshifted'))
    )


def create_poisson_evolution_joint(taxa, arg):
    joint_list = (
        [
            create_poisson_tree_likelihood('like', taxa, arg),
        ]
        + create_time_tree_prior(taxa, arg)
        + create_clock_prior(arg)
    )

    joint_dic = {
        'id': 'joint',
        'type': 'JointDistributionModel',
        'distributions': joint_list,
    }
    return joint_dic


def create_evolution_joint(taxa, alignment, arg):
    likelihood_dic = create_tree_likelihood('like', taxa, alignment, arg)
    prior_dic = {
        'id': 'prior',
        'type': 'JointDistributionModel',
        'distributions': create_evolution_priors(taxa, arg),
    }
    joint_dic = {
        'id': 'joint',
        'type': 'JointDistributionModel',
        'distributions': [
            likelihood_dic,
        ],
    }

    if len(prior_dic["distributions"]) > 0:
        joint_dic["distributions"].append(prior_dic)

    if arg.location_regex:
        data_type_location = create_general_data_type(
            'data_type.location', 'location', taxa
        )
        location_dic = create_tree_likelihood_general(
            'location', data_type_location, taxa, arg
        )
        joint_dic['distributions'].append(location_dic)
    elif arg.trait and arg.metadata:
        for trait in arg.trait:
            data_type_trait = create_general_data_type(
                f'data_type.{trait}', trait, taxa
            )
            trait_dic = create_tree_likelihood_general(
                trait, data_type_trait, taxa, arg
            )
            joint_dic['distributions'].append(trait_dic)

    return joint_dic


def get_engine(arg):
    """Import module or use cashed module if engine is specified in
    arguments."""
    evol = None
    if _engine is not None:
        evol = _engine
    elif arg.engine is not None:
        try:
            evol = importlib.import_module(arg.engine + '.cli.evolution')
        except ModuleNotFoundError as e:
            sys.stderr.write(str(e) + '\n')
            sys.stderr.write(
                f'The {arg.engine} engine does not exist or is not properly specified\n'
            )
            sys.exit(1)
    return evol
