from typing import List, Tuple

import numpy as np

from torchtree import Parameter
from torchtree.cli.evolution import (
    create_alignment,
    create_evolution_joint,
    create_evolution_parser,
    create_site_model_srd06_mus,
    create_taxa,
)
from torchtree.distributions import Distribution


def create_variational_parser(subprasers):
    parser = subprasers.add_parser(
        'advi', help='build a JSON file for variational inference'
    )
    create_evolution_parser(parser)

    parser.add_argument(
        '--iter',
        type=int,
        default=100000,
        help="""maximum number of iterations""",
    )
    parser.add_argument(
        '-q',
        '--variational',
        choices=['meanfield', 'fullrank'],
        default='meanfield',
        help="""variational distribution family""",
    )
    parser.add_argument(
        '-e',
        '--eta',
        default=0.1,
        type=float,
        help="""learning rate (default: 0.1)""",
    )
    parser.add_argument(
        '--elbo_samples',
        type=int,
        default=100,
        help="""number of samples for Monte Carlo estimate of ELBO""",
    )
    parser.add_argument(
        '--grad_samples',
        type=int,
        default=1,
        help="""number of samples for Monte Carlo estimate of gradients""",
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=1000,
        help="""number of samples to be drawn from the variational distribution""",
    )
    parser.add_argument(
        '--tol_rel_obj',
        type=float,
        default=0.01,
        help="""convergence tolerance on the relative norm of the objective
         (defaults: 0.001)""",
    )
    parser.add_argument(
        '--entropy',
        required=False,
        action='store_true',
        help="""use entropy instead of log Q in ELBO""",
    )
    parser.add_argument(
        '--distribution',
        required=False,
        choices=['Normal', 'LogNormal', 'Gamma'],
        default='Normal',
        help="""distribution for positive variable""",
    )
    parser.set_defaults(func=build_advi)
    return parser


def create_meanfield(
    var_id: str, json_object: dict, distribution: str
) -> Tuple[List[str], List[str]]:
    distributions = []
    var_parameters = []
    parameters = []
    if isinstance(json_object, list):
        for element in json_object:
            distrs, params = create_meanfield(var_id, element, distribution)
            distributions.extend(distrs)
            var_parameters.extend(params)
    elif isinstance(json_object, dict):
        if 'type' in json_object and json_object['type'] == 'Parameter':
            if 'lower' in json_object and 'upper' in json_object:
                if json_object['lower'] != json_object['upper']:
                    json_object['type'] = 'TransformedParameter'
                    json_object['transform'] = 'torch.distributions.SigmoidTransform'
                    json_object['x'] = {
                        'id': json_object['id'] + '.unres',
                        'type': 'Parameter',
                    }
                    if 'tensor' in json_object and isinstance(
                        json_object['tensor'], list
                    ):
                        json_object['x']['tensor'] = 0.5
                        json_object['x']['full'] = [len(json_object['tensor'])]
                        del json_object['tensor']
                    elif 'full' in json_object:
                        json_object['x']['tensor'] = 0.5
                        json_object['x']['full'] = json_object['full']
                        del json_object['tensor']
                        del json_object['full']

                    loc = Parameter.json_factory(
                        var_id + '.' + json_object['id'] + '.loc',
                        **{'full_like': json_object['id'], 'tensor': 0.5},
                    )

                    scale = {
                        'id': var_id + '.' + json_object['id'] + '.scale',
                        'type': 'TransformedParameter',
                        'transform': 'torch.distributions.ExpTransform',
                        'x': Parameter.json_factory(
                            var_id + '.' + json_object['id'] + '.scale.unres',
                            **{'full_like': json_object['id'], 'tensor': -1.89712},
                        ),
                    }
                    distr = Distribution.json_factory(
                        var_id + '.' + json_object['id'],
                        'torch.distributions.Normal',
                        json_object['id'] + '.unres',
                        {'loc': loc, 'scale': scale},
                    )
                    distributions.append(distr)
                    var_parameters.extend((loc['id'], scale['x']['id']))
                    parameters.append(json_object['id'])
            elif 'lower' in json_object:
                x_ref = json_object['id']
                if json_object['lower'] > 0:
                    json_object['type'] = 'TransformedParameter'
                    json_object['transform'] = 'torch.distributions.AffineTransform'
                    json_object['parameters'] = {
                        'loc': json_object['lower'],
                        'scale': 1.0,
                    }
                    json_object['x'] = {
                        'id': json_object['id'] + '.unshifted',
                        'type': 'Parameter',
                        'tensor': [0.5],
                    }
                    del json_object['tensor']
                    x_ref += '.unshifted'
                elif distribution == 'Normal':
                    json_object['type'] = 'TransformedParameter'
                    json_object['transform'] = 'torch.distributions.ExpTransform'
                    json_object['x'] = {
                        'id': json_object['id'] + '.unres',
                        'type': 'Parameter',
                        'tensor': np.log(json_object['tensor']).tolist(),
                    }

                    del json_object['tensor']
                    if 'full' in json_object:
                        json_object['x']['full'] = json_object['full']
                        del json_object['full']
                    elif 'full_like' in json_object:
                        json_object['x']['full_like'] = json_object['full_like']
                        del json_object['full_like']
                    x_ref += '.unres'

                loc = Parameter.json_factory(
                    var_id + '.' + x_ref + '.loc',
                    **{'full_like': json_object['id'], 'tensor': 0.5},
                )

                scale = {
                    'id': var_id + '.' + x_ref + '.scale',
                    'type': 'TransformedParameter',
                    'transform': 'torch.distributions.ExpTransform',
                    'x': Parameter.json_factory(
                        var_id + '.' + x_ref + '.scale.unres',
                        **{'full_like': json_object['id'], 'tensor': -1.89712},
                    ),
                }
                distr = Distribution.json_factory(
                    var_id + '.' + json_object['id'],
                    f'torch.distributions.{distribution}',
                    x_ref,
                    {'loc': loc, 'scale': scale},
                )
                distributions.append(distr)
                var_parameters.extend((loc['id'], scale['x']['id']))
                parameters.append(json_object['id'])
            elif 'simplex' in json_object and json_object['simplex']:
                json_object['type'] = 'TransformedParameter'
                json_object['transform'] = 'torch.distributions.StickBreakingTransform'
                if 'full' in json_object:
                    json_object['x'] = {
                        'id': json_object['id'] + '.unres',
                        'type': 'Parameter',
                        'tensor': 0.0,
                        'full': [json_object['full'][0] - 1],
                    }
                    del json_object['full']
                else:
                    json_object['x'] = {
                        'id': json_object['id'] + '.unres',
                        'type': 'Parameter',
                        'tensor': 0.0,
                        'full': [len(json_object['tensor']) - 1],
                    }
                del json_object['tensor']
                x_ref = json_object['id'] + '.unres'

                loc = Parameter.json_factory(
                    var_id + '.' + x_ref + '.loc',
                    **{'full_like': json_object['id'] + '.unres', 'tensor': 0.5},
                )

                scale = {
                    'id': var_id + '.' + x_ref + '.scale',
                    'type': 'TransformedParameter',
                    'transform': 'torch.distributions.ExpTransform',
                    'x': Parameter.json_factory(
                        var_id + '.' + json_object['id'] + '.scale.unres',
                        **{
                            'full_like': json_object['id'] + '.unres',
                            'tensor': -1.89712,
                        },
                    ),
                }
                distr = Distribution.json_factory(
                    var_id + '.' + json_object['id'],
                    'torch.distributions.Normal',
                    json_object['id'] + '.unres',
                    {'loc': loc, 'scale': scale},
                )
                distributions.append(distr)
                var_parameters.extend((loc['id'], scale['x']['id']))
                parameters.append(json_object['id'])
            else:
                loc = Parameter.json_factory(
                    var_id + '.' + json_object['id'] + '.loc',
                    **{'full_like': json_object['id'], 'tensor': 0.5},
                )

                scale = {
                    'id': var_id + '.' + json_object['id'] + '.scale',
                    'type': 'TransformedParameter',
                    'transform': 'torch.distributions.ExpTransform',
                    'x': Parameter.json_factory(
                        var_id + '.' + json_object['id'] + '.scale.unres',
                        **{'full_like': json_object['id'], 'tensor': -1.89712},
                    ),
                }
                distr = Distribution.json_factory(
                    var_id + '.' + json_object['id'],
                    'torch.distributions.Normal',
                    json_object['id'],
                    {'loc': loc, 'scale': scale},
                )
                distributions.append(distr)
                var_parameters.extend((loc['id'], scale['x']['id']))

        else:
            for value in json_object.values():
                distrs, params = create_meanfield(var_id, value, distribution)
                distributions.extend(distrs)
                var_parameters.extend(params)
    return distributions, var_parameters


def create_variational_model(id_, joint, arg) -> Tuple[dict, List[str]]:
    variational = {'id': id_, 'type': 'JointDistributionModel'}
    distributions, parameters = create_meanfield(id_, joint, arg.distribution)
    variational['distributions'] = distributions
    return variational, parameters


def create_advi(joint, variational, parameters, arg):
    advi_dic = {
        'id': 'advi',
        'type': 'Optimizer',
        'algorithm': 'torch.optim.Adam',
        'maximize': True,
        'lr': arg.eta,
        '_checkpoint': False,
        'iterations': arg.iter,
        'loss': {
            'id': 'elbo',
            'type': 'ELBO',
            'samples': arg.grad_samples,
            'joint': joint,
            'variational': variational,
        },
        'parameters': parameters,
    }

    advi_dic['convergence'] = {
        'type': 'StanVariationalConvergence',
        'max_iterations': arg.iter,
        'loss': 'elbo',
        'every': 100,
        'samples': arg.elbo_samples,
        'tol_rel_obj': arg.tol_rel_obj,
    }
    advi_dic['scheduler'] = {
        'type': 'torchtree.optim.Scheduler',
        'scheduler': 'torch.optim.lr_scheduler.LambdaLR',
        'lr_lambda': 'lambda epoch: 1.0 / (epoch + 1)**0.5',
    }
    return advi_dic


def create_sampler(id_, var_id, parameters, arg):
    return {
        "id": id_,
        "type": "Sampler",
        "model": var_id,
        "samples": arg.samples,
        "loggers": [
            {
                "id": "logger",
                "type": "Logger",
                "file_name": "samples.csv",
                "_parameters": [
                    "joint",
                    "varmodel",
                    "likelihood",
                    "gmrf",
                    "coalescent",
                ],
                "parameters": parameters,
                "delimiter": "\t",
            }
        ],
    }


def build_advi(arg):
    json_list = []
    taxa = create_taxa('taxa', arg)
    json_list.append(taxa)

    alignment = create_alignment('alignment', 'taxa', arg)
    json_list.append(alignment)

    jacobians_list = []
    if arg.clock is not None:
        if arg.heights == 'ratio':
            if arg.distribution == 'Normal':
                jacobians_list.append('tree.root_height')
            jacobians_list.extend(['tree', 'tree.ratios'])
        elif arg.heights == 'shift':
            if arg.distribution == 'Normal':
                jacobians_list.append('tree.shifts')

    if arg.model == 'SRD06':
        json_list.append(create_site_model_srd06_mus('srd06.mus'))

        for tag in ('12', '3'):
            jacobians_list.append(f'substmodel.{tag}' + '.frequencies')
    elif arg.model == 'HKY' or arg.model == 'GTR':
        jacobians_list.append('substmodel' + '.frequencies')
        if arg.model == 'GTR':
            jacobians_list.append('substmodel' + '.rates')

    if arg.clock == 'horseshoe':
        branch_model_id = 'branchmodel'
        jacobians_list.append(f'{branch_model_id}.rates.logdiff')
    if arg.categories > 1:
        if arg.distribution == 'Normal':
            jacobians_list.append('sitemodel.shape')
    if arg.coalescent in ('skyride', 'skygrid'):
        if arg.distribution == 'Normal':
            jacobians_list.append('gmrf.precision')

    joint_dic = create_evolution_joint(taxa, 'alignment', arg)
    joint_dic['distributions'].extend(jacobians_list)

    json_list.append(joint_dic)

    var_dic, var_parameters = create_variational_model('var', json_list, arg)
    json_list.append(var_dic)

    advi_dic = create_advi('joint', 'var', var_parameters, arg)
    json_list.append(advi_dic)

    parameters = []
    if arg.clock is not None:
        branch_model_id = 'branchmodel'
        if arg.heights == 'ratio':
            parameters.extend(["tree.ratios", "tree.root_height"])
        elif arg.heights == 'shift':
            parameters.extend(["tree.heights"])

        if arg.clock == 'ucln':
            parameters.extend(
                (
                    f'{branch_model_id}.rates.prior.mean',
                    f'{branch_model_id}.rates.prior.scale',
                )
            )
        else:
            parameters.append(f"{branch_model_id}.rate")

        if arg.clock == 'horseshoe' or arg.clock == 'ucln':
            parameters.append(f'{branch_model_id}.rates')
    else:
        parameters = ['tree.blens']

    if arg.coalescent is not None:
        parameters.append("coalescent.theta")
        if arg.coalescent in ('skygrid', 'skyride'):
            parameters.append('gmrf.precision')

    if arg.model == 'SRD06':
        for tag in ('12', '3'):
            parameters.extend(
                [f"substmodel.{tag}.kappa", f"substmodel.{tag}.frequencies"]
            )
    elif arg.model == 'GTR':
        parameters.extend(["substmodel.rates", "substmodel.frequencies"])
    elif arg.model == 'HKY':
        parameters.extend(["substmodel.kappa", "substmodel.frequencies"])

    if arg.model == 'SRD06':
        parameters.append("srd06.mus")

    if arg.categories > 1:
        if arg.model == 'SRD06':
            for tag in ('12', '3'):
                parameters.append(f"sitemodel.{tag}.shape")
        else:
            parameters.append("sitemodel.shape")

    json_list.append(create_sampler('sampler', 'var', parameters, arg))
    return json_list
