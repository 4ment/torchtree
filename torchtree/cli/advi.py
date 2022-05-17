from __future__ import annotations

import json
import logging
import sys
from typing import Union

import numpy as np
import torch

from torchtree import Parameter
from torchtree.cli.evolution import (
    create_alignment,
    create_evolution_joint,
    create_evolution_parser,
    create_poisson_evolution_joint,
    create_site_model_srd06_mus,
    create_taxa,
)
from torchtree.cli.jacobians import create_jacobians
from torchtree.distributions import Distribution

logger = logging.getLogger(__name__)


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
        '--lr',
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
        '--K_grad_samples',
        type=int,
        default=1,
        help="number of samples for Monte Carlo estimate of gradients"
        " using multisample objective",
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
    parser.add_argument(
        '--stem',
        required=False,
        help="""stem for output file""",
    )
    parser.add_argument(
        '--init_fullrank',
        required=False,
        help="""checkpoint file from a meanfield analysis""",
    )
    parser.add_argument(
        '--convergence_every',
        type=int,
        default=100,
        help="""convergence every N iterations""",
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


def create_tril(scales: torch.Tensor) -> torch.Tensor:
    """Create a 1 dimentional tensor containing a flatten tridiagonal matrix.

    A covariance matrix is created using scales**2 for variances and the covariances
    are set to zero. A tridiagonal is created using the cholesky decomposition and the
    diagonal elements are replaced with their log.

    :param scales: standard deviations
    :return:
    """
    dim = len(scales)
    cov = torch.full((dim, dim), 0.0)
    cov[range(dim), range(dim)] = scales * scales
    tril = torch.linalg.cholesky(cov)
    indices = torch.tril_indices(row=dim, col=dim, offset=0)
    tril[range(dim), range(dim)] = tril[range(dim), range(dim)].log()
    return tril[indices[0], indices[1]]


def create_fullrank_from_meanfield(params, path):
    with open(path) as fp:
        checkpoint = json.load(fp)
    locs = []
    log_scales = []
    for param in checkpoint:
        if '.loc' in param['id']:
            locs.append(param)
        elif '.scale.unres' in param['id']:
            log_scales.append(param)
        else:
            sys.stderr.write(param['id'])
            sys.exit(2)
    sorted(
        locs,
        key=lambda x: params.index(x['id'].replace('.loc', '').replace('var.', '')),
    )
    sorted(
        log_scales,
        key=lambda x: params.index(
            x['id'].replace('.scale.unres', '').replace('var.', '')
        ),
    )
    return locs, log_scales


def create_fullrank(var_id, json_object, arg):
    group_map = {}
    parameters = {'UNSPECIFIED': []}
    gather_parameters(json_object, group_map, parameters)
    res = apply_transforms_for_fullrank(var_id, parameters['UNSPECIFIED'])
    x = list(map(lambda x: x[1], res))

    if arg.init_fullrank:
        loc_list, log_scale_list = create_fullrank_from_meanfield(x, arg.init_fullrank)
        locs = torch.cat([torch.tensor(loc['tensor']) for loc in loc_list])
        scales = torch.cat(
            [torch.tensor(log_scale['tensor']).exp() for log_scale in log_scale_list]
        )
        tril = create_tril(scales)
    else:
        locs = torch.cat(list(map(lambda x: torch.tensor(x[2]), res)))
        tril = create_tril(torch.full(locs.shape, 0.001))

    distr = {
        'id': var_id,
        'type': 'MultivariateNormal',
        'x': x,
        'parameters': {
            'loc': {
                'id': f"{var_id}.loc",
                'type': 'Parameter',
                'tensor': locs.tolist(),
            },
            'scale_tril': {
                'id': f"{var_id}.scale_tril",
                'type': 'TransformedParameter',
                'transform': 'TrilExpDiagonalTransform',
                'x': {
                    'id': f"{var_id}.scale_tril.unres",
                    'type': 'Parameter',
                    'tensor': tril.tolist(),
                },
            },
        },
    }
    var_parameters = (f"{var_id}.loc", f"{var_id}.scale_tril.unres")
    return distr, var_parameters


def create_flexible_variational(arg, json_object):
    group_map = {}
    parameters = {'UNSPECIFIED': []}
    for i, distr in enumerate(arg.variational):
        if '(' in distr:
            var_type, rest = distr.split('(')
            var_id = _unique_id(var_type, group_map)
            parameters[var_id] = []
            for id_ in rest.rstrip(')').split(','):
                group_map[id_] = var_id
    gather_parameters(json_object, group_map, parameters)

    joint_var = []
    var_parameters = []
    for id_, params in parameters.items():
        if id_.lower().startswith('fullrank'):
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
        elif id_ in ('UNSPECIFIED', 'Normal', 'LogNormal', 'Gamma', 'Weibull'):
            distribution = (
                _unique_id('Normal', group_map) if id_ == 'UNSPECIFIED' else id_
            )
            for param in params:
                distr, var_params = create_meanfield(
                    f"var.{distribution}", param, distribution
                )
                joint_var.extend(distr)
                var_parameters.extend(var_params)
    return joint_var, var_parameters


def create_realnvp(var_id, json_object, arg):
    group_map = {}
    parameters = {'UNSPECIFIED': []}
    gather_parameters(json_object, group_map, parameters)
    res = apply_transforms_for_fullrank(var_id, parameters['UNSPECIFIED'])
    x = list(map(lambda x: x[1], res))

    params = torch.cat(list(map(lambda x: torch.tensor(x[2]), res)))

    distr = {
        "id": var_id,
        "type": "RealNVP",
        "x": x,
        "base": {
            "id": "base",
            "type": "Distribution",
            "distribution": "torchtree.distributions.Normal",
            "x": {"id": "dummy", "type": "Parameter", "zeros": params.shape[-1]},
            "parameters": {
                "loc": {
                    "id": f"{var_id}.base.loc",
                    "type": "Parameter",
                    "zeros": params.shape[-1],
                },
                "scale": {
                    "id": f"{var_id}.base.scale",
                    "type": "Parameter",
                    "ones": params.shape[-1],
                },
            },
        },
        "n_blocks": 2,
        "hidden_size": 2,
        "n_hidden": 1,
    }
    var_parameters = (var_id,)
    return distr, var_parameters


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


def apply_sigmoid_transformed(json_object, value=None):
    unres_id = json_object['id'] + '.unres'
    json_object['type'] = 'TransformedParameter'
    json_object['transform'] = 'torch.distributions.SigmoidTransform'
    json_object['x'] = {
        'id': unres_id,
        'type': 'Parameter',
    }
    if 'tensor' in json_object and isinstance(json_object['tensor'], list):
        if value is None:
            json_object['x']['tensor'] = (
                torch.distributions.SigmoidTransform()
                .inv(torch.tensor(json_object['tensor']))
                .tolist()
            )
        else:
            json_object['x']['tensor'] = value
            json_object['x']['full'] = [len(json_object['tensor'])]
        del json_object['tensor']
    elif 'full' in json_object:
        if value is None:
            json_object['x']['tensor'] = (
                torch.distributions.SigmoidTransform()
                .inv(torch.tensor(json_object['tensor']))
                .tolist()
            )
        else:
            json_object['x']['tensor'] = value
        json_object['x']['full'] = json_object['full']
        del json_object['tensor']
        del json_object['full']
    else:
        logger.debug(
            'apply_sigmoid_transformed only works on json object containing'
            ' tensor or full'
        )
        sys.stderr.write('error from apply_sigmoid_transformed\n')
        sys.stderr.write(json_object)
        sys.exit(1)
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
        'tensor': (np.array(json_object['tensor']) - loc).tolist(),
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
            'tensor': torch.distributions.StickBreakingTransform()
            .inv(torch.tensor(json_object['tensor']))
            .tolist(),
        }
    del json_object['tensor']
    return unres_id


def create_normal_distribution(var_id, x_unres, json_object, loc, scale):
    loc_param = Parameter.json_factory(
        var_id + '.' + x_unres + '.loc',
        **{'full_like': x_unres, 'tensor': loc},
    )

    if isinstance(loc, list):
        del loc_param['full_like']

    scale_param = {
        'id': var_id + '.' + x_unres + '.scale',
        'type': 'TransformedParameter',
        'transform': 'torch.distributions.ExpTransform',
        'x': Parameter.json_factory(
            var_id + '.' + x_unres + '.scale.unres',
            **{'full_like': x_unres, 'tensor': scale},
        ),
    }
    if isinstance(scale, list):
        del scale_param['x']['full_like']

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


def create_weibull_distribution(var_id, x_unres, json_object, scale, concentration):
    scale_param = {
        'id': var_id + '.' + json_object['id'] + '.scale',
        'type': 'TransformedParameter',
        'transform': 'torch.distributions.ExpTransform',
        'x': Parameter.json_factory(
            var_id + '.' + json_object['id'] + '.scale.unres',
            **{'full_like': json_object['id'], 'tensor': scale},
        ),
    }
    concentration_param = {
        'id': var_id + '.' + json_object['id'] + '.concentration',
        'type': 'TransformedParameter',
        'transform': 'torch.distributions.ExpTransform',
        'x': Parameter.json_factory(
            var_id + '.' + json_object['id'] + '.concentration.unres',
            **{'full_like': json_object['id'], 'tensor': concentration},
        ),
    }
    distr = Distribution.json_factory(
        var_id + '.' + json_object['id'],
        'torch.distributions.Weibull',
        x_unres,
        {'scale': scale_param, 'concentration': concentration_param},
    )
    return distr, scale_param, concentration_param


def create_meanfield(
    var_id: str, json_object: dict, distribution: str
) -> tuple[list[str], list[str]]:
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
                    apply_sigmoid_transformed(json_object)
                    distrs, params = create_meanfield(
                        var_id, json_object['x'], distribution
                    )
                    distributions.extend(distrs)
                    var_parameters.extend(params)
                    return distributions, var_parameters
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
                    tensor = np.array(json_object['tensor'])
                    loc = np.log(tensor / np.sqrt(1 + 0.001 / tensor**2)).tolist()
                    scale_log = np.log(
                        np.sqrt(np.log(1 + 0.001 / tensor**2))
                    ).tolist()
                    unres_id = apply_exp_transform(json_object)
                    distr, loc, scale = create_normal_distribution(
                        var_id, unres_id, json_object, loc, scale_log
                    )
                    var_parameters.extend((loc['id'], scale['x']['id']))
                elif distribution in ('Gamma', 'Weibull'):
                    if distribution == 'Gamma':
                        distr, concentration, rate = create_gamma_distribution(
                            var_id, json_object['id'], json_object, 0, 2.3
                        )
                    elif distribution == 'Weibull':
                        distr, scale, concentration = create_weibull_distribution(
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
                tensor = np.array(json_object['tensor'])
                distr, loc, scale = create_normal_distribution(
                    var_id,
                    json_object['id'],
                    json_object,
                    json_object['tensor'],
                    -1.89712,
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
    json_object: Union[dict, list],
) -> list[tuple[str, str, list]]:
    var_parameters = []
    if isinstance(json_object, list):
        for element in json_object:
            params = apply_transforms_for_fullrank(var_id, element)
            var_parameters.extend(params)
    elif isinstance(json_object, dict):
        if 'type' in json_object and json_object['type'] == 'Parameter':
            if 'lower' in json_object and 'upper' in json_object:
                if json_object['lower'] != json_object['upper']:
                    unres_id = apply_sigmoid_transformed(json_object)
                    tensor_list = json_object['x']['tensor']
                    # full is list representing the shape/length of the tensor
                    # and tensor is a float
                    if 'full' in json_object['x']:
                        tensor_list = (
                            json_object['x']['full'] * json_object['x']['tensor']
                        )
                    var_parameters.append((json_object['id'], unres_id, tensor_list))
            elif 'lower' in json_object:
                if json_object['lower'] > 0:
                    apply_affine_transform(json_object, json_object['lower'], 1.0)
                    unres_id = apply_exp_transform(json_object['x'])
                    tensor_list = json_object['x']['x']['tensor']
                    var_parameters.append((json_object['id'], unres_id, tensor_list))
                elif json_object['lower'] == 0:
                    unres_id = apply_exp_transform(json_object)
                    tensor_list = json_object['x']['tensor']
                    if 'full' in json_object['x']:
                        tensor_list = torch.full(
                            json_object['x']['full'], json_object['x']['tensor']
                        ).tolist()
                    var_parameters.append((json_object['id'], unres_id, tensor_list))
                else:
                    raise NotImplementedError
            elif 'simplex' in json_object and json_object['simplex']:
                unres_id = apply_simplex_transform(json_object)
                tensor_list = json_object['x']['tensor']
                if 'full' in json_object['x']:
                    tensor_list = torch.full(
                        json_object['x']['full'], json_object['x']['tensor']
                    ).tolist()
                var_parameters.append((json_object['id'], unres_id, tensor_list))
            else:
                tensor_list = json_object['tensor']
                if 'full' in json_object:
                    tensor_list = torch.full(
                        json_object['full'], json_object['tensor']
                    ).tolist()
                var_parameters.append(
                    (json_object['id'], json_object['id'], tensor_list)
                )

        else:
            for value in json_object.values():
                params = apply_transforms_for_fullrank(var_id, value)
                var_parameters.append(params)
    return var_parameters


def create_variational_model(id_, joint, arg) -> tuple[dict, list[str]]:
    variational = {'id': id_, 'type': 'JointDistributionModel'}
    if len(arg.variational) == 1 and arg.variational[0] == 'meanfield':
        distributions, parameters = create_meanfield(id_, joint, arg.distribution)
    elif (
        arg.variational == 'fullrank'
        or len(arg.variational) == 1
        and arg.variational[0] == 'fullrank'
    ):
        return create_fullrank(id_, joint, arg)
    elif len(arg.variational) == 1 and arg.variational[0] == 'realnvp':
        return create_realnvp(id_, joint, arg.distribution)
    else:
        distributions, parameters = create_flexible_variational(arg, joint)
    variational['distributions'] = distributions
    return variational, parameters


def create_advi(joint, variational, parameters, arg):
    if arg.K_grad_samples > 1:
        grad_samples = [arg.grad_samples, arg.K_grad_samples]
    else:
        grad_samples = arg.grad_samples

    if arg.stem:
        checkpoint = arg.stem + '-checkpoint.json'
    else:
        checkpoint = 'checkpoint.json'
    advi_dic = {
        'id': 'advi',
        'type': 'Optimizer',
        'algorithm': 'torch.optim.Adam',
        'options': {'lr': arg.lr},
        'maximize': True,
        'checkpoint': checkpoint,
        'iterations': arg.iter,
        'loss': {
            'id': 'elbo',
            'type': 'ELBO',
            'samples': grad_samples,
            'joint': joint,
            'variational': variational,
        },
        'parameters': parameters,
    }

    advi_dic['convergence'] = {
        'type': 'StanVariationalConvergence',
        'max_iterations': arg.iter,
        'loss': 'elbo',
        'every': arg.convergence_every,
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
    if arg.stem:
        file_name = arg.stem + '-samples.csv'
        tree_file_name = arg.stem + '-samples.trees'
    else:
        file_name = 'samples.csv'
        tree_file_name = 'samples.trees'

    parameters2 = list(filter(lambda x: 'tree.ratios' != x, parameters))

    return {
        "id": id_,
        "type": "Sampler",
        "model": var_id,
        "samples": arg.samples,
        "loggers": [
            {
                "id": "logger",
                "type": "Logger",
                "file_name": file_name,
                "parameters": ['joint', 'like', var_id] + parameters2,
                "delimiter": "\t",
            },
            {
                "id": "tree.logger",
                "type": "TreeLogger",
                "file_name": tree_file_name,
                "tree_model": "tree",
            },
        ],
    }


def build_advi(arg):
    json_list = []
    taxa = create_taxa('taxa', arg)
    json_list.append(taxa)

    if not arg.poisson:
        alignment = create_alignment('alignment', 'taxa', arg)
        json_list.append(alignment)

    if arg.model == 'SRD06':
        json_list.append(create_site_model_srd06_mus('srd06.mus'))

    if arg.poisson:
        joint_dic = create_poisson_evolution_joint(taxa, arg)
    else:
        joint_dic = create_evolution_joint(taxa, 'alignment', arg)
    json_list.append(joint_dic)

    # convert Parameters with constraints to TransformedParameters
    # and create variational distribution
    var_dic, var_parameters = create_variational_model('variational', json_list, arg)

    # extract references of TransformedParameters but not those coming from the
    # variational distribution
    jacobians_list = create_jacobians(json_list)
    if arg.clock is not None and arg.heights == 'ratio':
        jacobians_list.append('tree')
    if arg.coalescent in ('skygrid', 'skyride'):
        jacobians_list.remove("coalescent.theta")
    joint_dic['distributions'].extend(jacobians_list)

    json_list.append(var_dic)

    advi_dic = create_advi('joint', 'variational', var_parameters, arg)
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
        elif arg.coalescent == 'piecewise':
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
        json_list.append(create_sampler('sampler', 'variational', parameters, arg))
    return json_list
