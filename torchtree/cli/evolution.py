import re

from dendropy import Tree

from torchtree import Parameter, ViewParameter
from torchtree.cli.priors import create_one_on_x_prior
from torchtree.core.utils import process_object
from torchtree.distributions import Distribution
from torchtree.distributions.ctmc_scale import CTMCScale
from torchtree.distributions.scale_mixture import ScaleMixtureNormal
from torchtree.evolution.alignment import (
    Alignment,
    calculate_F3x4,
    read_fasta_sequences,
)
from torchtree.evolution.datatype import CodonDataType
from torchtree.evolution.taxa import Taxa, Taxon
from torchtree.evolution.tree_model import (
    ReparameterizedTimeTreeModel,
    UnRootedTreeModel,
)
from torchtree.evolution.tree_model_flexible import FlexibleTimeTreeModel


def create_evolution_parser(parser):
    parser.add_argument('-i', '--input', required=True, help="""alignment file""")
    parser.add_argument('-t', '--tree', required=True, help="""tree file""")
    parser.add_argument(
        '-m',
        '--model',
        choices=['JC69', 'HKY', 'GTR', 'SRD06'] + ['MG94'] + ['LG', 'WAG'],
        default='JC69',
        help="""substitution model [default: %(default)s]""",
    )
    parser.add_argument(
        '-I',
        '--invariant',
        required=False,
        action='store_true',
        help="""include a proportion of invariant sites""",
    )
    parser.add_argument(
        '-f',
        '--frequencies',
        required=False,
        help="""frequencies""",
    )
    parser.add_argument(
        '-C',
        '--categories',
        metavar='C',
        required=False,
        type=int,
        default=1,
        help="""number of rate categories""",
    )
    parser.add_argument(
        '--brlenspr',
        required=False,
        choices=['exponential', 'gammadir'],
        default='gammadir',
        help="""prior on branch lengths (unrooted tree)""",
    )
    parser.add_argument(
        '--clock',
        required=False,
        choices=['strict', 'ucln', 'horseshoe'],
        default=None,
        help="""type of clock""",
    )
    parser.add_argument(
        '--heights',
        required=False,
        choices=['ratio', 'shift'],
        default='ratio',
        help="""type of node height reparameterization""",
    )
    parser.add_argument(
        '--rate', required=False, type=float, help="""fixed substitution rate"""
    )
    parser.add_argument(
        '--dates',
        default=None,
        help="""regular expression to capture sampling date in sequence names""",
    )
    parser.add_argument(
        '--genetic_code',
        required=False,
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

    parser = add_coalescent(parser)

    parser = add_birth_death(parser)

    parser.add_argument(
        '--grid',
        type=int,
        help="""number of grid points (number of segments) for skygrid and BDSK""",
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
        choices=['constant', 'exponential', 'skyride', 'skygrid'],
        default=None,
        help="""type of coalescent""",
    )
    parser.add_argument(
        '--cutoff',
        type=float,
        help="""a cutoff for skygrid""",
    )
    parser.add_argument(
        '--time-aware',
        required=False,
        action='store_true',
        help="""use time aware GMRF""",
    )
    return parser


def create_tree_model(id_: str, taxa: dict, arg):
    tree_format = 'newick'
    with open(arg.tree) as fp:
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
        with open(arg.tree) as fp:
            newick = fp.read()
            newick = newick.strip()

    kwargs = {}
    if arg.keep:
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

            root_height['lower'] = offset
            tree_model = ReparameterizedTimeTreeModel.json_factory(
                id_, newick, ratios, root_height, 'taxa', **kwargs
            )
        elif arg.heights == 'shift':
            shifts = Parameter.json_factory(
                f'{id_}.shifts', **{'tensor': 0.1, 'full': [len(dates) - 1]}
            )
            shifts['lower'] = 0.0
            node_heights = {
                'id': f'{id_}.heights',
                'type': 'TransformedParameter',
                'transform': 'torchtree.evolution.tree_height_transform'
                '.DifferenceNodeHeightTransform',
                'x': shifts,
                'parameters': {'tree_model': id_},
            }
            tree_model = FlexibleTimeTreeModel.json_factory(
                id_, newick, node_heights, 'taxa', **kwargs
            )
    else:
        branch_lengths = Parameter.json_factory(
            f'{id_}.blens', **{'tensor': 0.1, 'full': [len(taxa['taxa']) * 2 - 3]}
        )
        branch_lengths['lower'] = 0.0
        tree_model = UnRootedTreeModel.json_factory(
            id_, newick, branch_lengths, 'taxa', **kwargs
        )
    return tree_model


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


def create_tree_likelihood(id_, taxa, alignment, arg):
    if arg.model == 'SRD06':
        branch_model = None
        branch_model_id = None
        tree_id = 'tree'
        tree_model = create_tree_model(tree_id, taxa, arg)
        if arg.clock is not None:
            branch_model_id = 'branchmodel'
            branch_model = create_branch_model(
                branch_model_id, tree_id, len(taxa['taxa']), arg
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

    if arg.clock is not None:
        treelikelihood_model['branch_model'] = create_branch_model(
            'branchmodel', tree_id, len(taxa['taxa']), arg
        )
    return treelikelihood_model


def create_site_model(id_, arg, w=None):
    if arg.categories == 1:
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


def create_branch_model(id_, tree_id, taxa_count, arg):
    if arg.rate is not None:
        rate = [arg.rate]
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


def create_substitution_model(id_, model, arg):
    if model == 'JC69':
        return {'id': id_, 'type': 'JC69'}
    elif model == 'HKY' or model == 'GTR':
        frequencies = Parameter.json_factory(
            f'{id_}.frequencies', **{'tensor': [0.25] * 4}
        )
        frequencies['simplex'] = True

        if arg.frequencies is not None and arg.frequencies != 'equal':
            frequencies['tensor'] = list(map(float, arg.frequencies.split(',')))
            if len(frequencies['tensor']) != 4:
                raise ValueError(
                    f'The dimension of the frequencies parameter '
                    f'({len(frequencies["tensor"])}) does not match the data type '
                    f'state count 4'
                )

        if model == 'HKY':
            kappa = Parameter.json_factory(f'{id_}.kappa', **{'tensor': [3.0]})
            kappa['lower'] = 0.0

            return {
                'id': id_,
                'type': 'HKY',
                'kappa': kappa,
                'frequencies': frequencies,
            }
        else:
            rates = Parameter.json_factory(
                f'{id_}.rates', **{'tensor': 1 / 6, 'full': [6]}
            )
            rates['simplex'] = True
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
                sequences = read_fasta_sequences(arg.input)
                taxa = Taxa(
                    'taxa', [Taxon(sequence.taxon, {}) for sequence in sequences]
                )
                alignment = Alignment('alignment', sequences, taxa, data_type)
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
    alignment = read_fasta_sequences(arg.input)
    taxa_list = []
    for sequence in alignment:
        taxa_list.append({'id': sequence.taxon, 'type': 'Taxon'})
    taxa = {'id': id_, 'type': 'Taxa', 'taxa': taxa_list}
    if arg.clock is not None:
        if arg.dates is not None and arg.dates == '0':
            for idx, taxon in enumerate(taxa_list):
                taxon['attributes'] = {'date': 0.0}
        else:
            regex_date = r'_(\d+\.?\d*)$'
            if arg.dates is not None:
                regex_date = arg.dates
            regex = re.compile(regex_date)
            for idx, taxon in enumerate(taxa_list):
                res = re.search(regex, taxon['id'])
                taxon['attributes'] = {'date': float(res.group(1))}
    return taxa


def create_birth_death(birth_death_id, tree_id, arg):
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


def create_coalesent(id_, tree_id, theta_id, arg, **kwargs):
    if arg.coalescent == 'constant':
        coalescent = {
            'id': id_,
            'type': 'ConstantCoalescentModel',
            'theta': theta_id,
            'tree_model': tree_id,
        }
    elif arg.coalescent == 'exponential':
        coalescent = {
            'id': id_,
            'type': 'ExponentialCoalescentModel',
            'theta': theta_id,
            'growth': kwargs.get('growth'),
            'tree_model': tree_id,
        }
    elif arg.coalescent == 'skygrid':
        coalescent = {
            'id': id_,
            'type': 'PiecewiseConstantCoalescentGridModel',
            'theta': theta_id,
            'tree_model': tree_id,
            'cutoff': arg.cutoff,
        }
    elif arg.coalescent == 'skyride':
        coalescent = {
            'id': id_,
            'type': 'PiecewiseConstantCoalescentModel',
            'theta': theta_id,
            'tree_model': tree_id,
        }
    return coalescent


def create_substitution_model_priors(substmodel_id, model):
    joint_list = []
    if model == 'HKY' or model == 'GTR':
        joint_list.append(
            Distribution.json_factory(
                f'{substmodel_id}.frequencies.prior',
                'torch.distributions.Dirichlet',
                f'{substmodel_id}.frequencies',
                {'concentration': [1.0] * 4},
            )
        )

        if model == 'GTR':
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


def create_evolution_priors(arg):
    tree_id = 'tree'
    joint_list = []
    if arg.clock is not None:
        branch_model_id = 'branchmodel'
        if arg.clock == 'strict' and arg.rate is None:
            joint_list.append(
                CTMCScale.json_factory(
                    f'{branch_model_id}.rate.prior', f'{branch_model_id}.rate', 'tree'
                )
            )
        elif arg.clock == 'ucln':
            joint_list.extend(create_ucln_prior(branch_model_id))
        elif arg.clock == 'horseshoe':
            log_diff = {
                'id': f'{branch_model_id}.rates.logdiff',
                'type': 'TransformedParameter',
                'transform': 'LogDifferenceRateTransform',
                'x': f'{branch_model_id}.rates.unscaled',
                'parameters': {'tree_model': tree_id},
            }
            global_scale = Parameter.json_factory(
                f'{branch_model_id}.global.scale', **{'tensor': [1.0]}
            )
            local_scale = Parameter.json_factory(
                f'{branch_model_id}.local.scales',
                **{'tensor': 1.0, 'full_like': f'{branch_model_id}.rates.unscaled'},
            )
            global_scale['lower'] = 0.0
            local_scale['lower'] = 0.0
            joint_list.append(
                ScaleMixtureNormal.json_factory(
                    f'{branch_model_id}.scale.mixture.prior',
                    log_diff,
                    0.0,
                    global_scale,
                    local_scale,
                )
            )
            joint_list.append(
                CTMCScale.json_factory(
                    f'{branch_model_id}.rate.prior', f'{branch_model_id}.rate', 'tree'
                )
            )
            for p in ('global.scale', 'local.scales'):
                joint_list.append(
                    Distribution.json_factory(
                        f'{branch_model_id}.{p}.prior',
                        'torch.distributions.Cauchy',
                        f'{branch_model_id}.{p}',
                        {'loc': 0.0, 'scale': 1.0},
                    )
                )

        coalescent_id = 'coalescent'
        if arg.coalescent == 'constant' or arg.coalescent == 'exponential':
            joint_list.append(
                create_one_on_x_prior(
                    f'{coalescent_id}.theta.prior', f'{coalescent_id}.theta'
                )
            )
        if arg.coalescent == 'exponential':
            joint_list.append(
                Distribution.json_factory(
                    f'{coalescent_id}.growth.prior',
                    'torch.distributions.Laplace',
                    f'{coalescent_id}.growth',
                    {
                        'loc': 0.0,
                        'scale': 1.0,
                    },
                )
            )
        elif arg.coalescent in ('skygrid', 'skyride'):
            gmrf = {
                'id': 'gmrf',
                'type': 'GMRF',
                'x': f'{coalescent_id}.theta.log',
                'precision': Parameter.json_factory(
                    'gmrf.precision',
                    **{'tensor': [0.1]},
                ),
            }
            if arg.time_aware:
                gmrf['tree_model'] = tree_id
            gmrf['precision']['lower'] = 0.0
            joint_list.append(gmrf)
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
        elif arg.birth_death == 'bdsk':
            birth_death_id = 'bdsk'
            joint_list.append(
                Distribution.json_factory(
                    f'{birth_death_id}.R.prior',
                    'torch.distributions.LogNormal',
                    f'{birth_death_id}.R',
                    {
                        'mean': 1.0,
                        'scale': 1.25,
                    },
                ),
            )
            joint_list.append(
                Distribution.json_factory(
                    f'{birth_death_id}.delta.prior',
                    'torch.distributions.LogNormal',
                    f'{birth_death_id}.delta',
                    {
                        'mean': 1.0,
                        'scale': 1.25,
                    },
                ),
            )
            joint_list.append(
                Distribution.json_factory(
                    f'{birth_death_id}.origin.prior',
                    'torch.distributions.LogNormal',
                    f'{birth_death_id}.origin.unshifted',
                    {
                        'mean': 1.0,
                        'scale': 1.25,
                    },
                ),
            )
            joint_list.append(
                Distribution.json_factory(
                    f'{birth_death_id}.rho.prior',
                    'torch.distributions.Beta',
                    f'{birth_death_id}.rho',
                    {
                        'concentration1': 1.0,
                        'concentration0': 9999.0,
                    },
                ),
            )
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


def create_evolution_joint(taxa, alignment, arg):
    joint_list = []
    joint_list.append(create_tree_likelihood('like', taxa, alignment, arg))
    joint_approx_dic = joint_list.copy()
    params = {}

    if arg.coalescent is not None:
        coalescent_id = 'coalescent'
        if arg.coalescent in ('constant', 'exponential'):
            theta = Parameter.json_factory(
                f'{coalescent_id}.theta', **{'tensor': [3.0]}
            )
            theta['lower'] = 0.0
        if arg.coalescent == 'exponential':
            growth = Parameter.json_factory(
                f'{coalescent_id}.growth', **{'tensor': [1.0]}
            )
            params['growth'] = growth
        elif arg.coalescent == 'skygrid':
            theta_log = Parameter.json_factory(
                f'{coalescent_id}.theta.log', **{'tensor': 3.0, 'full': [arg.grid]}
            )
            theta = {
                'id': f'{coalescent_id}.theta',
                'type': 'TransformedParameter',
                'transform': 'torch.distributions.ExpTransform',
                'x': theta_log,
            }
        elif arg.coalescent == 'skyride':
            theta_log = Parameter.json_factory(
                f'{coalescent_id}.theta.log',
                **{'tensor': 3.0, 'full': [len(taxa['taxa']) - 1]},
            )
            theta = {
                'id': f'{coalescent_id}.theta',
                'type': 'TransformedParameter',
                'transform': 'torch.distributions.ExpTransform',
                'x': theta_log,
            }
        joint_list.append(create_coalesent(coalescent_id, 'tree', theta, arg, **params))
        joint_approx_dic.append(joint_list[-1])
    elif arg.birth_death is not None:
        birth_death_id = 'bdsk'
        joint_list.append(create_birth_death(birth_death_id, 'tree', arg))
        joint_approx_dic.append(joint_list[-1])

    prior_list = create_evolution_priors(arg)

    joint_dic = {
        'id': 'joint',
        'type': 'JointDistributionModel',
        'distributions': joint_list + prior_list,
    }
    return joint_dic
