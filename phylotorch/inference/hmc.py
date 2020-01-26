import torch
from torch.distributions import Normal
import numpy as np

# Adaptation p.15 No-U-Turn samplers Algo 5
def adaptation(rho, t, step_size_init, H_t, eps_bar, desired_accept_rate=0.8):
    # rho is current acceptance ratio
    # t is current iteration
    t = t + 1
    if util.has_nan_or_inf(torch.tensor([rho])):
        alpha = 0  # Acceptance rate is zero if nan.
    else:
        alpha = min(1., float(torch.exp(torch.FloatTensor([rho]))))
    mu = float(torch.log(10 * torch.FloatTensor([step_size_init])))
    gamma = 0.05
    t0 = 10
    kappa = 0.75
    H_t = (1 - (1 / (t + t0))) * H_t + (1 / (t + t0)) * (desired_accept_rate - alpha)
    x_new = mu - (t ** 0.5) / gamma * H_t
    step_size = float(torch.exp(torch.FloatTensor([x_new])))
    x_new_bar = t ** -kappa * x_new + (1 - t ** -kappa) * torch.log(torch.FloatTensor([eps_bar]))
    eps_bar = float(torch.exp(x_new_bar))
    # import pdb; pdb.set_trace()
    # print('rho: ',rho)
    # print('alpha: ',alpha)
    # print('step_size: ',step_size)
    # adapt_stepsize_list.append(torch.exp(x_new_bar))
    return step_size, eps_bar, H_t


def leapfrog(params, func, steps, step_size):
    x0 = params
    p0 = Normal(torch.zeros(params.size(), requires_grad=False),
                                    torch.ones(params.size(), requires_grad=False)).sample()
    params = params.detach().requires_grad_()
    U = func(params)
    U.backward()
    dU = -params.grad
    # Half step for momentum
    pStep = p0 - step_size / 2.0 * dU

    # Full step for position
    xStep = x0 + step_size * pStep

    for _ in range(steps):
        params = xStep.detach().requires_grad_()
        U = func(params)
        U.backward()
        dU = -params.grad
        # Update momentum
        pStep -= step_size * dU

        # Update position
        xStep += step_size * pStep

    params = xStep
    # Half for momentum
    pStep -= step_size / 2 * dU
    return params, torch.dot(p0, p0) / 2 - torch.dot(pStep, pStep) / 2


def hmc(beta, func, iters=range(1000), steps=20, step_size=0.001, logger=None):
    with torch.no_grad():
        logP = func(beta)
    samples = []
    probs = []
    accept = 0

    for epoch in iters:
        beta_new, kinetic_diff = leapfrog(beta, func, steps, step_size)
        with torch.no_grad():
            proposed_logP = func(beta_new)

        alpha = proposed_logP - logP + kinetic_diff

        if alpha >= 0 or alpha > np.log(np.random.uniform()):
            beta = beta_new
            logP = proposed_logP
            accept += 1
        samples.append(np.copy(beta.detach().numpy()))
        if logger is not None:
            logger(epoch, logP.item(), accept)
        probs.append(logP.item())
    return samples, probs