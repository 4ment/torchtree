from torchtree.cli.evolution import (
    create_alignment,
    create_evolution_joint,
    create_evolution_parser,
    create_site_model_srd06_mus,
    create_taxa,
)
from torchtree.cli.jacobians import create_jacobians
from torchtree.cli.map import make_unconstrained


def create_hmc_parser(subprasers):
    parser = subprasers.add_parser('hmc', help='build a JSON file for HMC inference')
    create_evolution_parser(parser)

    parser.add_argument(
        '--iter',
        type=int,
        default=100000,
        help="""maximum number of iterations""",
    )
    parser.add_argument(
        '--step_size',
        default=0.01,
        type=float,
        help="""step size for leafrog integrator""",
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=10,
        help="""number of Step size for leafrog integrator""",
    )
    parser.add_argument(
        '--every',
        type=int,
        default=1000,
        help="""logging frequency of samples""",
    )
    parser.add_argument(
        '--stem',
        required=False,
        help="""stem for output file""",
    )
    parser.set_defaults(func=build_hmc)
    return parser


def create_hmc(joint, parameters, parameters_unres, arg):
    hmc_json = {
        "id": "hmc",
        "type": "HMC",
        "joint": joint,
        "iterations": arg.iter,
        "steps": arg.steps,
        "step_size": arg.step_size,
        "parameters": parameters_unres,
        "step_size_adaptor": True,
    }
    if arg.stem:
        hmc_json["loggers"] = [
            {
                "id": "logger",
                "type": "Logger",
                "parameters": parameters,
                "delimiter": "\t",
                "file_name": f"{arg.stem}.csv",
                "every": 100,
            }
        ]
    return hmc_json


def build_hmc(arg):
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

    opt_dict = create_hmc('joint', parameters, parameters_unres, arg)
    json_list.append(opt_dict)

    jacobians_list = create_jacobians(json_list)
    if arg.clock is not None and arg.heights == 'ratio':
        jacobians_list.append('tree')
    if arg.coalescent in ('skygrid', 'skyride'):
        jacobians_list.remove("coalescent.theta")
    joint_dic['distributions'].extend(jacobians_list)

    return json_list
