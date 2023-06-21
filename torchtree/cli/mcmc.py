from __future__ import annotations

from torchtree.cli import PLUGIN_MANAGER
from torchtree.cli.evolution import (
    create_alignment,
    create_evolution_joint,
    create_evolution_parser,
    create_site_model_srd06_mus,
    create_taxa,
)
from torchtree.cli.jacobians import create_jacobians
from torchtree.cli.loggers import create_loggers
from torchtree.cli.operators import create_sliding_window_operator
from torchtree.cli.utils import make_unconstrained


def create_mcmc_parser(subprasers):
    parser = subprasers.add_parser("mcmc", help="build a JSON file for MCMC inference")
    create_evolution_parser(parser)

    parser.add_argument(
        "--iter",
        type=int,
        default=100000,
        help="""maximum number of iterations [default: %(default)d]""",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=1000,
        help="""logging frequency of samples [default: %(default)d]""",
    )

    parser.add_argument(
        "--target_acc_prob",
        type=float,
        default=0.8,
        help="""target acceptance probability [default: %(default)f]""",
    )
    parser.add_argument(
        "--stem",
        required=True,
        help="""stem for output file""",
    )
    parser.set_defaults(func=build_mcmc)
    return parser


def create_mcmc(joint, parameters, parameters_unres, arg):
    hmc_json = {
        "id": "hmc",
        "type": "MCMC",
        "joint": joint,
        "iterations": arg.iter,
        "operators": [],
    }

    for param in parameters_unres:
        operator = create_sliding_window_operator(param["id"], joint, param, arg)
        hmc_json["operators"].append(operator)

    if arg.stem:
        hmc_json["loggers"] = create_loggers(parameters, arg)

    return hmc_json


def build_mcmc(arg):
    json_list = []
    taxa = create_taxa("taxa", arg)
    json_list.append(taxa)

    alignment = create_alignment("alignment", "taxa", arg)
    json_list.append(alignment)

    if arg.model == "SRD06":
        json_list.append(create_site_model_srd06_mus("srd06.mus"))

    joint_dic = create_evolution_joint(taxa, "alignment", arg)

    json_list.append(joint_dic)

    parameters_unres, parameters = make_unconstrained(json_list)

    jacobians_list = create_jacobians(json_list)
    if arg.clock is not None and arg.heights == "ratio":
        jacobians_list.append("tree")
    if arg.coalescent in ("skygrid", "skyride"):
        jacobians_list.remove("coalescent.theta")

    joint_jacobian = {
        "id": "joint.jacobian",
        "type": "JointDistributionModel",
        "distributions": ["joint"] + jacobians_list,
    }
    json_list.append(joint_jacobian)

    opt_dict = create_mcmc("joint.jacobian", parameters, parameters_unres, arg)
    json_list.append(opt_dict)

    for plugin in PLUGIN_MANAGER.plugins():
        plugin.process_all(arg, json_list)

    return json_list
