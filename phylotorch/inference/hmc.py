import numpy as np
import torch
from torch.distributions import Normal


def leapfrog(params, func, steps, step_size):
    x0 = params
    p0 = Normal(
        torch.zeros(params.size(), requires_grad=False),
        torch.ones(params.size(), requires_grad=False),
    ).sample()
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
