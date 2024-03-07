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
from torchtree.cli.utils import make_unconstrained


def create_hmc_parser(subprasers):
    parser = subprasers.add_parser("hmc", help="build a JSON file for HMC inference")
    create_evolution_parser(parser)

    parser.add_argument(
        "--iter",
        type=int,
        default=100000,
        help="""maximum number of iterations [default: %(default)d]""",
    )
    parser.add_argument(
        "--step_size",
        default=0.01,
        type=float,
        help="""step size for leapfrog integrator [default: %(default)f]""",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="""number of steps for leapfrog integrator [default: %(default)d]""",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=1000,
        help="""logging frequency of samples [default: %(default)d]""",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="""number of iterations for warmup [default: %(default)d]""",
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
    parser.add_argument(
        "--mass_matrix",
        choices=["diagonal", "dense"],
        default="diagonal",
        help="""mass matrix type [default: %(default)s]""",
    )
    parser.add_argument(
        "--adapt_mass_matrix",
        action="store_true",
        help="""adapt mass matrix""",
    )
    parser.add_argument(
        "--adapt_step_size",
        choices=["dualaveraging", "adaptive"],
        help="""adapt step size""",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="""one parameter per operator""",
    )
    parser.add_argument(
        "--join",
        required=False,
        help="""multiple parameters per operator""",
    )
    parser.set_defaults(func=build_hmc)
    return parser


def create_stan_windowed_adaptation(joint, parameters, parameters_unres, arg):
    stan_windowed_adaptation = {
        "id": "adaptor",
        "type": "StanWindowedAdaptation",
        "warmup": 1000,
        "initial_window": 75,
        "final_window": 50,
        "base_window": 25,
        "step_size_adaptor": {
            "id": "step_size",
            "type": "StepSizeAdaptation",
            "mu": 0.5,
            "delta": 0.5,
            "gamma": 0.05,
            "kappa": 0.75,
            "t0": 10,
        },
        "mass_matrix_adaptor": {
            "id": "matrix_adaptor",
            "type": "DiagonalMassMatrixAdaptor",
            "parameters": parameters_unres,
        },
    }
    return stan_windowed_adaptation


def create_hmc_operator(id_, joint, parameters, arg):
    mass_matrix = {
        "id": f"{id_}.mass.matrix",
        "type": "Parameter",
    }

    if isinstance(parameters, list) and len(parameters) > 1:
        parameter_ids = []
        count = 0
        for param in parameters:
            parameter_ids.append(param["id"])
            if "tensor" in param:
                if isinstance(param["tensor"], list):
                    count += len(param["tensor"])
                elif "full" in param:
                    count += param["full"][0]
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        if arg.mass_matrix == "diagonal":
            mass_matrix["ones"] = count
        elif arg.mass_matrix == "dense":
            mass_matrix["eye"] = count
    else:
        if isinstance(parameters, list) and len(parameters) == 1:
            parameters = parameters[0]
        parameter_ids = parameters["id"]
        if arg.mass_matrix == "diagonal":
            mass_matrix["ones_like"] = parameters["id"]
        elif arg.mass_matrix == "dense":
            mass_matrix["eye_like"] = parameters["id"]

    operator = {
        "id": f"{id_}.operator",
        "type": "HMCOperator",
        "joint": joint,
        "parameters": parameter_ids,
        "weight": 1.0,
        "integrator": {
            "id": f"{id_}.leapfrog",
            "type": "LeapfrogIntegrator",
            "steps": arg.steps,
            "step_size": arg.step_size,
        },
        "mass_matrix": mass_matrix,
        "adaptors": [],
    }

    if arg.adapt_mass_matrix:
        operator["adaptors"].append(
            {
                "id": f"{id_}.mass.matrix.adaptor",
                "type": "MassMatrixAdaptor",
                "mass_matrix": f"{id_}.mass.matrix",
                "update_frequency": 10,
                "parameters": parameter_ids,
            }
        )
    if arg.adapt_step_size == "dualaveraging":
        operator["adaptors"].append(
            {
                "id": f"{id_}.step.size.adaptor",
                "type": "DualAveragingStepSize",
                "integrator": f"{id_}.leapfrog",
            }
        )
    elif arg.adapt_step_size == "adaptive":
        operator["adaptors"].append(
            {
                "id": f"{id_}.step.size.adaptor",
                "type": "AdaptiveStepSize",
                "integrator": f"{id_}.leapfrog",
                "target_acceptance_probability": arg.target_acc_prob,
                # "use_acceptance_rate": True,
            }
        )
    return operator


def create_hmc(joint, parameters, parameters_unres, arg):
    hmc_json = {
        "id": "hmc",
        "type": "MCMC",
        "joint": joint,
        "iterations": arg.iter,
        "operators": [],
    }
    if arg.split or arg.join is not None:
        if arg.join is not None:
            groups = arg.join.split(":")
            dic = {}
            dic2 = {group: {} for group in groups}
            for param in parameters_unres:
                for group in groups:
                    if param["id"] in group:
                        dic[param["id"]] = param
                        dic2[group][param["id"]] = 1
                if param["id"] not in dic:
                    operator = create_hmc_operator(param["id"], joint, param, arg)
                    hmc_json["operators"].append(operator)
            for group in dic2:
                params = group.split(",")
                operator = create_hmc_operator(
                    params[0], joint, [dic[p] for p in params], arg
                )
                hmc_json["operators"].append(operator)
        else:
            for param in parameters_unres:
                operator = create_hmc_operator(param["id"], joint, param, arg)
                hmc_json["operators"].append(operator)
    else:
        operator = create_hmc_operator("hmc", joint, parameters_unres, arg)
        hmc_json["operators"].append(operator)

    if arg.warmup > 0:
        hmc_json["adaptation"] = create_stan_windowed_adaptation(
            joint, parameters, [param["id"] for param in parameters_unres], arg
        )

    if arg.stem:
        hmc_json["loggers"] = create_loggers(parameters, arg)

    return hmc_json


def build_hmc(arg):
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
    if arg.coalescent in ("skygrid", "skyride") or arg.coalescent.startswith(
        "piecewise"
    ):
        jacobians_list.remove("coalescent.theta")

    joint_jacobian = {
        "id": "joint.jacobian",
        "type": "JointDistributionModel",
        "distributions": ["joint"] + jacobians_list,
    }
    json_list.append(joint_jacobian)

    opt_dict = create_hmc("joint.jacobian", parameters, parameters_unres, arg)
    json_list.append(opt_dict)

    for plugin in PLUGIN_MANAGER.plugins():
        plugin.process_all(arg, json_list)

    return json_list
