from torchtree import Parameter
from torchtree.distributions import Distribution
from torchtree.distributions.ctmc_scale import CTMCScale
from torchtree.distributions.scale_mixture import ScaleMixtureNormal


def create_one_on_x_prior(id_, theta):
    return {
        'id': id_,
        'type': 'Distribution',
        'distribution': 'OneOnX',
        'x': theta,
    }


def create_clock_horseshoe_prior(branch_model_id, tree_id):
    prior_list = []
    log_diff = {
        'id': f'{branch_model_id}.rates.logdiff',
        'type': 'TransformedParameter',
        'transform': 'LogDifferenceRateTransform',
        'x': f'{branch_model_id}.rates.unscaled',
        'parameters': {'tree_model': tree_id},
    }
    global_scale = Parameter.json_factory(
        f'{branch_model_id}.global.scale', **{'tensor': [1.0]}
    )
    local_scale = Parameter.json_factory(
        f'{branch_model_id}.local.scales',
        **{'tensor': 1.0, 'full_like': f'{branch_model_id}.rates.unscaled'},
    )
    global_scale['lower'] = 0.0
    local_scale['lower'] = 0.0
    prior_list.append(
        ScaleMixtureNormal.json_factory(
            f'{branch_model_id}.scale.mixture.prior',
            log_diff,
            0.0,
            global_scale,
            local_scale,
        )
    )
    prior_list.append(
        CTMCScale.json_factory(
            f'{branch_model_id}.rate.prior', f'{branch_model_id}.rate', 'tree'
        )
    )
    for p in ('global.scale', 'local.scales'):
        prior_list.append(
            Distribution.json_factory(
                f'{branch_model_id}.{p}.prior',
                'torch.distributions.Cauchy',
                f'{branch_model_id}.{p}',
                {'loc': 0.0, 'scale': 1.0},
            )
        )
    return prior_list
