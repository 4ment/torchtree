from __future__ import annotations

from typing import Union

import torch

from torchtree.cli.evolution import (
    create_alignment,
    create_evolution_joint,
    create_evolution_parser,
    create_site_model_srd06_mus,
    create_taxa,
)


def create_map_parser(subprasers):
    parser = subprasers.add_parser(
        'map', help='build a JSON file for maximum a posteriori inference'
    )
    create_evolution_parser(parser)

    parser.add_argument(
        '--lr',
        default=1.0,
        type=float,
        help="""learning rate""",
    )
    parser.add_argument(
        '--max_iter',
        type=int,
        default=20,
        help="""maximal number of iterations per optimization step (default: 20)""",
    )
    parser.add_argument(
        '--max_eval',
        type=int,
        help="""maximal number of function evaluations per optimization step
        (default: max_iter * 1.25)""",
    )
    parser.add_argument(
        '--tolerance_grad',
        type=float,
        default=1e-5,
        help="""termination tolerance on first order optimality (default: 1e-5)""",
    )
    parser.add_argument(
        '--tolerance_change',
        type=float,
        default=1e-9,
        help="""termination tolerance on function value/parameter changes
        (default: 1e-9)""",
    )
    parser.add_argument(
        '--history_size',
        type=int,
        default=100,
        help="""update history size (default: 100)""",
    )
    parser.add_argument(
        '--line_search_fn',
        type=str,
        help="""either 'strong_wolfe' or None (default: None)""",
    )
    parser.add_argument(
        '--stem',
        required=True,
        help="""stem for output files""",
    )
    parser.set_defaults(func=build_optimizer)
    return parser


def make_unconstrained(json_object: Union[dict, list]) -> tuple[list[str], list[str]]:
    parameters = []
    parameters_unres = []
    if isinstance(json_object, list):
        for element in json_object:
            params_unres, params = make_unconstrained(element)
            parameters_unres.extend(params_unres)
            parameters.extend(params)
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
                    transform = torch.distributions.SigmoidTransform()
                    if 'tensor' in json_object and isinstance(
                        json_object['tensor'], list
                    ):
                        json_object['x']['tensor'] = transform.inv(
                            torch.tensor(json_object['tensor'])
                        ).tolist()
                    elif 'full' in json_object:
                        json_object['x']['tensor'] = transform.inv(
                            torch.tensor(json_object['tensor'])
                        ).item()
                        json_object['x']['full'] = json_object['full']
                        del json_object['full']
                    elif 'full_like' in json_object:
                        json_object['x']['tensor'] = transform.inv(
                            torch.tensor(json_object['tensor'])
                        ).item()
                        json_object['x']['full_like'] = json_object['full_like']
                        del json_object['full_like']
                    del json_object['tensor']

                    parameters.append(json_object['id'])
                    parameters_unres.append(json_object['x']['id'])
            elif 'lower' in json_object:
                if json_object['lower'] > 0:
                    json_object['type'] = 'TransformedParameter'
                    json_object['transform'] = 'torch.distributions.AffineTransform'
                    json_object['parameters'] = {
                        'loc': json_object['lower'],
                        'scale': 1.0,
                    }
                    transform = torch.distributions.AffineTransform(
                        json_object['lower'], 1.0
                    )
                    new_value = (
                        transform.inv(torch.tensor(json_object['tensor']))
                    ).tolist()

                    json_object['x'] = {
                        'id': json_object['id'] + '.unshifted',
                        'type': 'Parameter',
                        'tensor': new_value,
                        'lower': 0.0,
                    }
                    del json_object['tensor']

                    params_unres, params = make_unconstrained(json_object['x'])
                    parameters_unres.extend(params_unres)
                    parameters.extend(params)
                else:
                    json_object['type'] = 'TransformedParameter'
                    json_object['transform'] = 'torch.distributions.ExpTransform'

                    json_object['x'] = {
                        'id': json_object['id'] + '.unres',
                        'type': 'Parameter',
                    }
                    transform = torch.distributions.ExpTransform()

                    if 'full' in json_object:
                        json_object['x']['tensor'] = transform.inv(
                            torch.tensor(json_object['tensor'])
                        ).item()
                        json_object['x']['full'] = json_object['full']
                        del json_object['full']
                    elif 'full_like' in json_object:
                        json_object['x']['tensor'] = transform.inv(
                            torch.tensor(json_object['tensor'])
                        ).item()
                        json_object['x']['full_like'] = json_object['full_like']
                        del json_object['full_like']
                    else:
                        json_object['x']['tensor'] = transform.inv(
                            torch.tensor(json_object['tensor'])
                        ).tolist()

                    del json_object['tensor']

                    parameters.append(json_object['id'])
                    parameters_unres.append(json_object['x']['id'])
            elif 'simplex' in json_object and json_object['simplex']:
                json_object['type'] = 'TransformedParameter'
                json_object['transform'] = 'torch.distributions.StickBreakingTransform'
                transform = torch.distributions.StickBreakingTransform()
                if 'full' in json_object:
                    tensor_unres = transform.inv(
                        torch.full(json_object['full'], json_object['tensor'])
                    ).tolist()
                else:
                    tensor_unres = transform.inv(
                        torch.tensor(json_object['tensor'])
                    ).tolist()

                json_object['x'] = {
                    'id': json_object['id'] + '.unres',
                    'type': 'Parameter',
                    'tensor': tensor_unres,
                }
                del json_object['tensor']
                if 'full' in json_object:
                    del json_object['full']

                parameters.append(json_object['id'])
                parameters_unres.append(json_object['x']['id'])
            else:
                parameters.append(json_object['id'])
                parameters_unres.append(json_object['id'])

        else:
            for element in json_object.values():
                params_unres, params = make_unconstrained(element)
                parameters.extend(params)
                parameters_unres.extend(params_unres)
    return parameters_unres, parameters


def create_optimizer(joint, parameters, arg):
    return {
        "id": "bfgs",
        "type": "Optimizer",
        "algorithm": "torch.optim.LBFGS",
        "options": {"lr": arg.lr},
        "maximize": True,
        "iterations": 10,
        "max_iter": arg.max_iter,
        "loss": joint,
        "parameters": parameters,
    }


def create_logger(id_, parameters, arg):
    return {
        "id": id_,
        "type": "Logger",
        "parameters": parameters,
        "file_name": arg.stem + '.csv',
    }


def create_tree_logger(id_, tree_id, arg):
    return {
        "id": id_,
        "type": "TreeLogger",
        "tree_model": tree_id,
        "file_name": arg.stem + '.tree',
    }


def build_optimizer(arg):
    json_list = []
    taxa = create_taxa('taxa', arg)
    json_list.append(taxa)

    alignment = create_alignment('alignment', 'taxa', arg)
    json_list.append(alignment)

    if arg.model == 'SRD06':
        json_list.append(create_site_model_srd06_mus('srd06.mus'))

    joint_dic = create_evolution_joint(taxa, 'alignment', arg)

    json_list.append(joint_dic)

    parameters_unres, parameters = make_unconstrained(json_list)

    opt_dict = create_optimizer('joint', parameters_unres, arg)
    json_list.append(opt_dict)

    logger_dict = create_logger('logger', parameters, arg)
    tree_logger_dict = create_tree_logger('tree.logger', 'tree', arg)
    json_list.extend((logger_dict, tree_logger_dict))

    return json_list
