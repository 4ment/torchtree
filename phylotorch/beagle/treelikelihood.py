import torch
import torch.autograd
import numpy as np
from .tree import NodeHeightAutogradFunction


class TreeLikelihoodBeagle(object):
    def __init__(self, inst, bounds, pre_indexing, post_indexing, subst_model):
        self.inst = inst
        self.bounds = bounds
        self.pre_indexing = pre_indexing
        self.post_indexing = post_indexing
        self.subst_model = subst_model

    def log_prob(self, ratios_root_height, clock=None, susbt_rates=None, subst_freqs=None):
        treelike = TreelikelihoodAutogradFunction.apply
        log_P = treelike(self.inst, ratios_root_height, clock, susbt_rates, subst_freqs)
        # node_heights = transform_ratios(root_height, ratios, node_bounds, indexing)
        node_heights = NodeHeightAutogradFunction.apply(self.inst, ratios_root_height)
        return log_P, node_heights


class TreelikelihoodAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inst, branch_lengths, clock=None, rates=None, frequencies=None, weibull_shape=None):
        ctx.inst = inst
        ctx.rooted = True if clock is not None else False
        ctx.weibull = True if weibull_shape is not None else False

        phylo_model_param_block_map = inst.get_phylo_model_param_block_map()

        if clock is not None:
            # Set ratios in tree
            inst.tree_collection.trees[0].set_node_heights_via_height_ratios(branch_lengths.detach().numpy())
            # Set clock rate in tree
            inst_rates = np.array(inst.tree_collection.trees[0].rates, copy=False)
            inst_rates[:] = clock.detach().numpy()
            # libsbn does not use phylo_model_param_block_map for clock rates
            # phylo_model_param_block_map["clock rate"][:] = clock.detach().numpy()
        else:
            inst_branch_lengths = np.array(inst.tree_collection.trees[0].branch_lengths, copy=False)
            # branch_lengths[:] = input_branch_lengths.detach().numpy()
            inst_branch_lengths[:-1] = branch_lengths.detach().numpy()

        if weibull_shape is not None:
            phylo_model_param_block_map["Weibull shape"][:] = weibull_shape.detach().numpy()

        if rates is not None:
            phylo_model_param_block_map["GTR rates"][:] = rates.detach().numpy()
            phylo_model_param_block_map["frequencies"][:] = frequencies.detach().numpy()

        log_prob = np.array(inst.log_likelihoods())[0]
        return torch.tensor(log_prob, dtype=torch.float64)

    @staticmethod
    def backward(ctx, grad_output):
        # index, = ctx.saved_tensors
        gradient = ctx.inst.phylo_gradients()[0]
        if ctx.rooted:
            branch_grad = torch.tensor(np.array(gradient.ratios_root_height)) * grad_output
            clock_rate_grad = torch.tensor(np.array(gradient.clock_model)) * grad_output
        else:
            # branch_grad = torch.tensor(np.array(gradient.branch_lengths)[:-1])*grad_output
            branch_grad = torch.tensor(np.array(gradient.branch_lengths)[:-2])*grad_output
            clock_rate_grad = None
        if ctx.weibull:
            weibull_grad = torch.tensor(np.array(gradient.site_model))*grad_output
        else:
            weibull_grad = None
        return None, branch_grad, clock_rate_grad, None, None, weibull_grad


def libsn_likelihood(inst, node_bounds, indexing, rates, frequencies,
                     root_height, ratios, clock, weibull_shape, subst_rates, subst_freqs):
    treelike = TreelikelihoodAutogradFunction.apply
    log_P = treelike(inst, torch.cat((ratios, root_height)), clock, rates, frequencies, weibull_shape)
    # node_heights = transform_ratios(root_height, ratios, node_bounds, indexing)
    node_heights = NodeHeightAutogradFunction.apply(inst, torch.cat((ratios, root_height)))
    return log_P, node_heights
