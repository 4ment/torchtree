from __future__ import annotations

from torchtree.cli.evolution import COALESCENT_PIECEWISE


def create_loggers(parameters: list[str], arg) -> dict:
    models = ["joint.jacobian", "joint", "like", "prior"]
    if arg.coalescent:
        models.append("coalescent")
        if arg.coalescent in COALESCENT_PIECEWISE and not arg.gmrf_integrated:
            models.append('gmrf')
    return [
        {
            "id": "logger",
            "type": "Logger",
            "parameters": models + parameters,
            "delimiter": "\t",
            "file_name": f"{arg.stem}.csv",
            "every": arg.log_every,
        },
        {
            "id": "looger.trees",
            "type": "TreeLogger",
            "tree_model": "tree",
            "file_name": f"{arg.stem}.trees",
            "every": arg.log_every,
        },
    ]
