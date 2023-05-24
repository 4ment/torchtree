from torchtree.cli import PLUGIN_MANAGER
from torchtree.cli.evolution import (
    create_alignment,
    create_evolution_joint,
    create_evolution_parser,
    create_site_model_srd06_mus,
    create_taxa,
)
from torchtree.cli.utils import make_unconstrained


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
        "parameters": [parameter["id"] for parameter in parameters],
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

    for plugin in PLUGIN_MANAGER.plugins():
        plugin.process_all(arg, json_list)

    return json_list
