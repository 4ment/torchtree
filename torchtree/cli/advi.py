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
        nargs='*',
        # choices=['meanfield', 'fullrank'],
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


def _unique_id(id_, dic):
    index = 0
    unique_id = id_
    while unique_id in dic:
        index += 1
        unique_id = id_ + '.' + str(index)
    return unique_id


def create_flexible_variational(arg, json_object):
    group_map = {}
    parameters = {'UNSPECIFIED': []}
    for i, distr in enumerate(arg.variational):
        print(distr, distr.split('('))
        var_type, rest = distr.split('(')
        var_id = _unique_id(var_type, group_map)
        parameters[var_id] = []
        for id_ in rest.rstrip(')').split(','):
            group_map[id_] = var_id
    gather_parameters(json_object, group_map, parameters)

    joint_var = []
    var_parameters = []
    for id_, params in parameters.items():
        if id_.startswith('Fullrank'):
            res = apply_transforms_for_fullrank(id_, params)
            distr = {
                'id': f"var.{id_}",
                'type': 'MultivariateNormal',
                'x': list(map(lambda x: x[0], res)),
                'parameters': {
                    'loc': id_ + '.' + 'loc',
                    'scale_tril': id_ + '.' + 'scale_tril',
                },
            }
            var_parameters.extend(
                (distr['parameters']['loc'], distr['parameters']['scale_tril'])
            )
            joint_var.append(distr)
        elif id_ in ('UNSPECIFIED', 'Normal', 'LogNormal', 'Gamma'):
            distribution = (
                _unique_id('Normal', group_map) if id_ == 'UNSPECIFIED' else id_
            )
            for param in params:
                distr, var_params = create_meanfield(
                    f"var.{distribution}", param, distribution
                )
                joint_var.append(distr)
                var_parameters.extend(var_params)
    return joint_var, var_parameters


def gather_parameters(json_object: dict, group_map: dict, parameters: dict):
    if isinstance(json_object, list):
        for element in json_object:
            gather_parameters(element, group_map, parameters)
    elif isinstance(json_object, dict):
        if 'type' in json_object and json_object['type'] == 'Parameter':
            if (
                'lower' not in json_object
                or 'upper' not in json_object
                or json_object['lower'] != json_object['upper']
            ):
                if json_object['id'] in group_map:
                    parameters[group_map[json_object['id']]].append(json_object)
                else:
                    parameters['UNSPECIFIED'].append(json_object)
        else:
            for element in json_object.values():
                gather_parameters(element, group_map, parameters)


def apply_sigmoid_transformed(json_object, value=0.0):
    unres_id = json_object['id'] + '.unres'
    json_object['type'] = 'TransformedParameter'
    json_object['transform'] = 'torch.distributions.SigmoidTransform'
    json_object['x'] = {
        'id': unres_id,
        'type': 'Parameter',
    }
    if 'tensor' in json_object and isinstance(json_object['tensor'], list):
        json_object['x']['tensor'] = value
        json_object['x']['full'] = [len(json_object['tensor'])]
        del json_object['tensor']
    elif 'full' in json_object:
        json_object['x']['tensor'] = value
        json_object['x']['full'] = json_object['full']
        del json_object['tensor']
        del json_object['full']
    return unres_id


def apply_affine_transform(json_object, loc, scale):
    unshifted_id = json_object['id'] + '.unshifted'
    json_object['type'] = 'TransformedParameter'
    json_object['transform'] = 'torch.distributions.AffineTransform'
    json_object['parameters'] = {
        'loc': loc,
        'scale': scale,
    }
    json_object['x'] = {
        'id': unshifted_id,
        'type': 'Parameter',
        'tensor': [0.5],
    }
    json_object['x']['lower'] = 0
    del json_object['tensor']
    return unshifted_id


def apply_exp_transform(json_object):
    unres_id = json_object['id'] + '.unres'
    json_object['type'] = 'TransformedParameter'
    json_object['transform'] = 'torch.distributions.ExpTransform'
    json_object['x'] = {
        'id': unres_id,
        'type': 'Parameter',
        'tensor': np.log(json_object['tensor']).tolist(),
    }
    if 'full' in json_object:
        json_object['x']['full'] = json_object['full']
        del json_object['full']
    elif 'full_like' in json_object:
        json_object['x']['full_like'] = json_object['full_like']
        del json_object['full_like']
    del json_object['tensor']
    return unres_id


def apply_simplex_transform(json_object):
    unres_id = json_object['id'] + '.unres'
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
            'id': unres_id,
            'type': 'Parameter',
            'tensor': 0.0,
            'full': [len(json_object['tensor']) - 1],
        }
    del json_object['tensor']
    return unres_id


def create_normal_distribution(var_id, x_unres, json_object, loc, scale):
    loc_param = Parameter.json_factory(
        var_id + '.' + json_object['id'] + '.loc',
        **{'full_like': x_unres, 'tensor': loc},
    )

    scale_param = {
        'id': var_id + '.' + json_object['id'] + '.scale',
        'type': 'TransformedParameter',
        'transform': 'torch.distributions.ExpTransform',
        'x': Parameter.json_factory(
            var_id + '.' + json_object['id'] + '.scale.unres',
            **{'full_like': x_unres, 'tensor': scale},
        ),
    }
    distr = Distribution.json_factory(
        var_id + '.' + json_object['id'],
        'torch.distributions.Normal',
        x_unres,
        {'loc': loc_param, 'scale': scale_param},
    )
    return distr, loc_param, scale_param


def create_gamma_distribution(var_id, x_unres, json_object, concentration, rate):
    concentration_param = {
        'id': var_id + '.' + json_object['id'] + '.concentration',
        'type': 'TransformedParameter',
        'transform': 'torch.distributions.ExpTransform',
        'x': Parameter.json_factory(
            var_id + '.' + json_object['id'] + '.concentration.unres',
            **{'full_like': json_object['id'], 'tensor': concentration},
        ),
    }
    rate_param = {
        'id': var_id + '.' + json_object['id'] + '.rate',
        'type': 'TransformedParameter',
        'transform': 'torch.distributions.ExpTransform',
        'x': Parameter.json_factory(
            var_id + '.' + json_object['id'] + '.rate.unres',
            **{'full_like': json_object['id'], 'tensor': rate},
        ),
    }
    distr = Distribution.json_factory(
        var_id + '.' + json_object['id'],
        'torch.distributions.Gamma',
        x_unres,
        {'concentration': concentration_param, 'rate': rate_param},
    )
    return distr, concentration_param, rate_param


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
                    unres_id = apply_sigmoid_transformed(json_object, 0.5)
                    distr, loc, scale = create_normal_distribution(
                        var_id, unres_id, json_object, 0.5, -1.89712
                    )
                    distributions.append(distr)
                    var_parameters.extend((loc['id'], scale['x']['id']))
                    parameters.append(json_object['id'])
            elif 'lower' in json_object:
                if json_object['lower'] > 0:
                    apply_affine_transform(json_object, json_object['lower'], 1.0)

                    # now id becomes id.unshifted with a lower bound of 0 so another
                    # round of create_meanfield to create a id.unshifted.unres
                    # parameter or keep id.unshifted with a lognormal or gamma
                    # distribution
                    distrs, params = create_meanfield(
                        var_id, json_object['x'], distribution
                    )
                    distributions.extend(distrs)
                    var_parameters.extend(params)
                    return distributions, var_parameters
                elif distribution == 'Normal':
                    unres_id = apply_exp_transform(json_object)
                    distr, loc, scale = create_normal_distribution(
                        var_id, unres_id, json_object, 0.5, -1.89712
                    )
                    var_parameters.extend((loc['id'], scale['x']['id']))
                elif distribution == 'Gamma':
                    distr, concentration, rate = create_gamma_distribution(
                        var_id, json_object['id'], json_object, 0, 2.3
                    )
                    var_parameters.extend(
                        [p['x']['id'] for p in distr['parameters'].values()]
                    )
                distributions.append(distr)
                parameters.append(json_object['id'])
            elif 'simplex' in json_object and json_object['simplex']:
                unres_id = apply_simplex_transform(json_object)
                distr, loc, scale = create_normal_distribution(
                    var_id, unres_id, json_object, 0.5, -1.89712
                )
                distributions.append(distr)
                var_parameters.extend((loc['id'], scale['x']['id']))
                parameters.append(json_object['id'])
            else:
                distr, loc, scale = create_normal_distribution(
                    var_id, json_object['id'], json_object, 0.5, -1.89712
                )
                distributions.append(distr)
                var_parameters.extend((loc['id'], scale['x']['id']))

        else:
            for value in json_object.values():
                distrs, params = create_meanfield(var_id, value, distribution)
                distributions.extend(distrs)
                var_parameters.extend(params)
    return distributions, var_parameters


def apply_transforms_for_fullrank(
    var_id: str,
    json_object: [dict, list],
) -> List[Tuple[str, str, int]]:
    var_parameters = []
    if isinstance(json_object, list):
        for element in json_object:
            params = apply_transforms_for_fullrank(var_id, element)
            var_parameters.extend(params)
    elif isinstance(json_object, dict):
        if 'type' in json_object and json_object['type'] == 'Parameter':
            if 'lower' in json_object and 'upper' in json_object:
                if json_object['lower'] != json_object['upper']:
                    unres_id = apply_sigmoid_transformed(json_object, 0.5)
                    var_parameters.append((json_object['id'], unres_id, 1))
            elif 'lower' in json_object:
                if json_object['lower'] > 0:
                    apply_affine_transform(json_object, json_object['lower'], 1.0)
                    unres_id = apply_exp_transform(json_object['x'])
                    var_parameters.append((json_object['id'], unres_id, 1))
                elif json_object['lower'] == 0:
                    unres_id = apply_exp_transform(json_object)
                    var_parameters.append((json_object['id'], unres_id, 1))
                else:
                    raise NotImplementedError
            elif 'simplex' in json_object and json_object['simplex']:
                unres_id = apply_simplex_transform(json_object)
                var_parameters.append((json_object['id'], unres_id, 1))
            else:
                var_parameters.append((json_object['id'], json_object['id'], 1))

        else:
            for value in json_object.values():
                params = apply_transforms_for_fullrank(var_id, value)
                var_parameters.append(params)
    return var_parameters


def create_variational_model(id_, joint, arg) -> Tuple[dict, List[str]]:
    variational = {'id': id_, 'type': 'JointDistributionModel'}
    if arg.variational == 'meanfield':
        distributions, parameters = create_meanfield(id_, joint, arg.distribution)
    else:
        distributions, parameters = create_flexible_variational(arg, joint)
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
                "parameters": parameters,
                "delimiter": "\t",
            }
        ],
    }


def create_jacobians(json_object):
    params = []
    if isinstance(json_object, list):
        for element in json_object:
            params.extend(create_jacobians(element))
    elif isinstance(json_object, dict):
        if 'type' in json_object and json_object['type'] == 'TransformedParameter':
            if not (
                json_object['transform'] == 'torch.distributions.AffineTransform'
                and json_object['parameters']['scale'] == 1.0
            ):
                params.append(json_object['id'])
        for value in json_object.values():
            params.extend(create_jacobians(value))
    return params


def build_advi(arg):
    json_list = []
    taxa = create_taxa('taxa', arg)
    json_list.append(taxa)

    alignment = create_alignment('alignment', 'taxa', arg)
    json_list.append(alignment)

    if arg.model == 'SRD06':
        json_list.append(create_site_model_srd06_mus('srd06.mus'))

    joint_dic = create_evolution_joint(taxa, 'alignment', arg)
    json_list.append(joint_dic)

    # convert Parameters with constraints to TransformedParameters
    # and create variational distribution
    var_dic, var_parameters = create_variational_model('var', json_list, arg)

    # extract references of TransformedParameters but not those coming from the
    # variational distribution
    jacobians_list = create_jacobians(json_list)
    if arg.clock is not None and arg.heights == 'ratio':
        jacobians_list.append('tree')
    if arg.coalescent in ('skygrid', 'skyride'):
        jacobians_list.remove("coalescent.theta")
    joint_dic['distributions'].extend(jacobians_list)

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
        elif arg.coalescent == 'exponential':
            parameters.append('coalescent.growth')
    elif arg.birth_death is not None:
        parameters.append("bdsk.R")
        parameters.append("bdsk.delta")
        parameters.append("bdsk.rho")
        parameters.append("bdsk.origin")

    if arg.model == 'SRD06':
        for tag in ('12', '3'):
            parameters.extend(
                [f"substmodel.{tag}.kappa", f"substmodel.{tag}.frequencies"]
            )
    elif arg.model == 'GTR':
        parameters.extend(["substmodel.rates", "substmodel.frequencies"])
    elif arg.model == 'HKY':
        parameters.extend(["substmodel.kappa", "substmodel.frequencies"])
    elif arg.model == 'MG94':
        parameters.extend([f"substmodel.{p}" for p in ("kappa", "alpha", "beta")])

    if arg.model == 'SRD06':
        parameters.append("srd06.mus")

    if arg.categories > 1:
        if arg.model == 'SRD06':
            for tag in ('12', '3'):
                parameters.append(f"sitemodel.{tag}.shape")
        else:
            parameters.append("sitemodel.shape")

    if arg.samples > 0:
        json_list.append(create_sampler('sampler', 'var', parameters, arg))
    return json_list
