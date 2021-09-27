import re

from phylotorch import Parameter
from phylotorch.cli.priors import create_one_on_x_prior
from phylotorch.distributions import Distribution
from phylotorch.distributions.ctmc_scale import CTMCScale
from phylotorch.evolution.alignment import read_fasta_sequences
from phylotorch.evolution.tree_model import (
    ReparameterizedTimeTreeModel,
    UnRootedTreeModel,
)


def create_evolution_parser(parser):
    parser.add_argument('-i', '--input', required=True, help="""alignment file""")
    parser.add_argument('-t', '--tree', required=True, help="""tree file""")
    parser.add_argument(
        '-m',
        '--model',
        choices=['JC69', 'HKY', 'GTR'],
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
        '-C',
        '--categories',
        metavar='C',
        required=False,
        type=int,
        default=1,
        help="""number of rate categories""",
    )
    parser.add_argument(
        '--clock',
        required=False,
        choices=[
            'strict',
        ],
        default=None,
        help="""type of clock""",
    )

    parser.add_argument(
        '--rate', required=False, type=float, help="""substitution rate"""
    )
    parser.add_argument(
        '--dates',
        default=None,
        help="""regular expression to capture sampling date in sequence names""",
    )
    parser.add_argument(
        '--keep', action="store_true", help="""use branch length as starting values"""
    )

    parser = add_coalescent(parser)

    return parser


def add_coalescent(parser):
    parser.add_argument(
        '--coalescent',
        choices=['constant', 'skyride', 'skygrid'],
        default=None,
        help="""type of coalescent (constant or skyride)""",
    )
    parser.add_argument(
        '--grid',
        metavar='I',
        type=int,
        help="""Number of grid points in skygrid""",
    )
    parser.add_argument(
        '--cutoff',
        metavar='G',
        type=float,
        help="""a cutoff for skygrid""",
    )
    return parser


def create_tree_model(id_: str, taxa: dict, arg):
    with open(arg.tree) as fp:
        newick = fp.read()
        newick = newick.strip()

    kwargs = {}
    if arg.keep:
        kwargs['keep_branch_lengths'] = True

    if arg.clock is not None:
        dates = [taxon['attributes']['date'] for taxon in taxa['taxa']]
        offset = max(dates) - min(dates)

        ratios = Parameter.json_factory(
            f'{id_}.ratios', **{'tensor': 0.1, 'full': [len(dates) - 2]}
        )
        ratios['lower'] = 0.0
        ratios['upper'] = 1.0

        root_height = Parameter.json_factory(
            f'{id_}.root_height', **{'tensor': [offset + 1.0]}
        )

        if offset != 0.0:
            root_height['lower'] = offset
        tree_model = ReparameterizedTimeTreeModel.json_factory(
            id_, newick, ratios, root_height, 'taxa', **kwargs
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


def create_tree_likelihood(id_, taxa, arg):
    site_pattern = create_site_pattern('patterns', arg)
    site_model = create_site_model('sitemodel', arg)
    substitution_model = create_substitution_model('substmodel', arg)
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
    if arg.clock is not None:
        treelikelihood_model['branch_model'] = create_branch_model(
            'branchmodel', tree_id, arg
        )
    return treelikelihood_model


def create_site_model(id_, arg):
    if arg.categories == 1:
        return {'id': id_, 'type': 'ConstantSiteModel'}
    else:
        shape = Parameter.json_factory(f'{id_}.shape', **{'tensor': [0.1]})
        shape['lower'] = 0.0
        return {
            'id': id_,
            'type': 'WeibullSiteModel',
            'categories': arg.categories,
            'shape': shape,
        }


def create_branch_model(id_, tree_id, arg):
    if arg.rate is not None:
        rate = [arg.rate]
    else:
        rate = [0.001]
    rate_parameter = Parameter.json_factory(f'{id_}.rate', **{'tensor': rate})
    rate_parameter['lower'] = 0.0

    return {
        'id': id_,
        'type': 'StrictClockModel',
        'tree_model': tree_id,
        'rate': rate_parameter,
    }


def create_substitution_model(id_, arg):
    if arg.model == 'JC69':
        return {'id': id_, 'type': 'JC69'}
    elif arg.model == 'HKY' or arg.model == 'GTR':
        frequencies = Parameter.json_factory(
            f'{id_}.frequencies', **{'tensor': 0.25, 'full': [4]}
        )
        frequencies['simplex'] = True

        if arg.model == 'HKY':
            kappa = Parameter.json_factory(f'{id_}.kappa', **{'tensor': [1.0]})
            kappa['lower'] = 0.0

            return {
                'id': id_,
                'type': 'HKY',
                'kappa': kappa,
                'frequencies': frequencies,
            }
        else:
            rates = Parameter.json_factory(
                f'{id_}.rates', **{'tensor': 1 / 6, 'full': [6], 'simplex': True}
            )
            rates['simplex'] = True
            return {
                'id': id_,
                'type': 'GTR',
                'rates': rates,
                'frequencies': frequencies,
            }


def create_site_pattern(id_, arg):
    alignment = create_alignment(arg)
    site_pattern = {'id': id_, 'type': 'SitePattern', 'alignment': alignment}
    return site_pattern


def create_alignment(arg):
    sequences = read_fasta_sequences(arg.input)
    sequence_list = []
    for sequence in sequences:
        sequence_list.append({'taxon': sequence.taxon, 'sequence': sequence.sequence})
    alignment = {
        'id': 'alignment',
        'type': 'Alignment',
        'datatype': 'nucleotide',
        'taxa': 'taxa',
        'sequences': sequence_list,
    }
    return alignment


def create_taxa(id_, arg):
    alignment = read_fasta_sequences(arg.input)
    taxa_list = []
    for sequence in alignment:
        taxa_list.append({'id': sequence.taxon, 'type': 'Taxon'})
    taxa = {'id': id_, 'type': 'Taxa', 'taxa': taxa_list}
    if arg.dates is not None and float(arg.dates) == 0:
        for idx, taxon in enumerate(taxa_list):
            taxon['attributes'] = {'date': 0.0}
    elif arg.clock is not None:
        regex_date = r'_(\d+\.?\d*)$'
        if arg.dates is not None:
            regex_date = arg.dates
        regex = re.compile(regex_date)
        for idx, taxon in enumerate(taxa_list):
            res = re.search(regex, taxon['id'])
            taxon['attributes'] = {'date': float(res.group(1))}
    return taxa


def create_coalesent(id_, tree_id, theta_id, arg):
    if arg.coalescent == 'constant':
        coalescent = {
            'id': id_,
            'type': 'ConstantCoalescentModel',
            'theta': theta_id,
            'tree_model': tree_id,
        }
    return coalescent


def create_evolution_priors(arg):
    joint_list = []
    if arg.clock is not None:
        if arg.clock == 'strict':
            branch_model_id = 'branchmodel'
            joint_list.append(
                CTMCScale.json_factory(
                    f'{branch_model_id}.rate.prior', f'{branch_model_id}.rate', 'tree'
                )
            )

        if arg.coalescent == 'constant':
            coalescent_id = 'coalescent'
            joint_list.append(
                create_one_on_x_prior(
                    f'{coalescent_id}.theta.prior', f'{coalescent_id}.theta'
                )
            )

    if arg.model == 'HKY' or arg.model == 'GTR':
        substmodel_id = 'substmodel'
        joint_list.append(
            Distribution.json_factory(
                f'{substmodel_id}.frequencies.prior',
                'torch.distributions.Dirichlet',
                f'{substmodel_id}.frequencies',
                {'concentration': [1.0] * 4},
            )
        )

        if arg.model == 'GTR':
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

    if arg.categories > 1:
        sitemodel_id = 'sitemodel'
        joint_list.append(
            Distribution.json_factory(
                f'{sitemodel_id}.shape.prior',
                'torch.distributions.Exponential',
                f'{sitemodel_id}.shape',
                {'rate': 2.0},
            )
        )
    return joint_list


def create_evolution_joint(taxa, arg):
    joint_list = []
    joint_list.append(create_tree_likelihood('like', taxa, arg))
    joint_approx_dic = joint_list.copy()

    if arg.coalescent is not None:
        coalescent_id = 'coalescent'
        theta = Parameter.json_factory(f'{coalescent_id}.theta', **{'tensor': [3.0]})
        theta['lower'] = 0
        joint_list.append(create_coalesent(f'{coalescent_id}', 'tree', theta, arg))
        joint_approx_dic.append(joint_list[-1])

    prior_list = create_evolution_priors(arg)

    joint_dic = {
        'id': 'joint',
        'type': 'JointDistributionModel',
        'distributions': joint_list + prior_list,
    }
    return joint_dic
