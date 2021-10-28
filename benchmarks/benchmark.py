#!/usr/bin/env python

import argparse
from timeit import default_timer as timer
from typing import List, Tuple

import torch

from torchtree import Parameter
from torchtree.evolution.alignment import Alignment, Sequence
from torchtree.evolution.coalescent import ConstantCoalescent
from torchtree.evolution.datatype import NucleotideDataType
from torchtree.evolution.io import read_tree, read_tree_and_alignment
from torchtree.evolution.site_pattern import compress_alignment
from torchtree.evolution.substitution_model import JC69
from torchtree.evolution.taxa import Taxa, Taxon
from torchtree.evolution.tree_likelihood import (
    calculate_treelikelihood_discrete,
    calculate_treelikelihood_discrete_rescaled,
)
from torchtree.evolution.tree_model import (
    ReparameterizedTimeTreeModel,
    heights_from_branch_lengths,
)


def benchmark(f):
    def timed(replicates, *args):
        start = timer()
        for _ in range(replicates):
            out = f(*args)
        end = timer()
        total_time = end - start
        return total_time, out

    return timed


def log_prob_squashed(
    theta: torch.Tensor,
    node_heights: torch.Tensor,
    counts: torch.Tensor,
    taxa_count: int,
) -> torch.Tensor:
    heights_sorted, indices = torch.sort(node_heights, descending=False)
    counts_sorted = torch.gather(counts, -1, indices)
    lineage_count = counts_sorted.cumsum(-1)[..., :-1]

    durations = heights_sorted[..., 1:] - heights_sorted[..., :-1]
    lchoose2 = lineage_count * (lineage_count - 1) / 2.0
    return torch.sum(-lchoose2 * durations / theta, -1, keepdim=True) - (
        taxa_count - 1
    ) * torch.log(theta)


def log_prob(node_heights: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    taxa_shape = node_heights.shape[:-1] + (int((node_heights.shape[-1] + 1) / 2),)
    node_mask = torch.cat(
        [
            torch.full(taxa_shape, 0, dtype=torch.int),
            torch.full(
                taxa_shape[:-1] + (taxa_shape[-1] - 1,),
                1,
                dtype=torch.int,
            ),
        ],
        dim=-1,
    )

    indices = torch.argsort(node_heights, descending=False)
    heights_sorted = torch.gather(node_heights, -1, indices)
    node_mask_sorted = torch.gather(node_mask, -1, indices)
    lineage_count = torch.where(
        node_mask_sorted == 1,
        torch.full_like(theta, -1),
        torch.full_like(theta, 1),
    ).cumsum(-1)[..., :-1]

    durations = heights_sorted[..., 1:] - heights_sorted[..., :-1]
    lchoose2 = lineage_count * (lineage_count - 1) / 2.0
    return torch.sum(-lchoose2 * durations / theta, -1, keepdim=True) - (
        taxa_shape[-1] - 1
    ) * torch.log(theta)


def calculate_treelikelihood_discrete_split(
    tip_partials: torch.Tensor,
    weights: torch.Tensor,
    post_indexing: torch.Tensor,
    mats: torch.Tensor,
    freqs: torch.Tensor,
    props: torch.Tensor,
) -> torch.Tensor:
    r"""Calculate log tree likelihood

    number of tips: T
    number of internal nodes: I=T-1
    number of branches: B=2T-2
    number of states: S
    number of sites: N
    number of rate categories: K

    tip_partials is split into a list of T tensors [S,N] and I empty tensors are
    appended.
    The shape of internal partials after peeling is [...,K,S,N]

    :param tip_partials: tensors of partials [T,S,N]
    :param weights: [N]
    :param post_indexing: list of indexes in postorder
    :param mats: tensor of probability matrices [...,B,K,S,S]
    :param freqs: tensor of frequencies [...,1,S]
    :param props: tensor of proportions [...,K,1,1]
    :return: tree log likelihood [batch]
    """
    partials = list(torch.split(tip_partials, 1))
    partials.extend([torch.empty(0)] * (tip_partials.shape[0] - 1))

    for indices in post_indexing:
        left = int(indices[1].item())
        right = int(indices[2].item())
        partials[indices[0]] = (mats[..., left, :, :, :] @ partials[left]) * (
            mats[..., right, :, :, :] @ partials[right]
        )
    return torch.sum(
        torch.log(freqs @ torch.sum(props * partials[post_indexing[-1][0]], -3))
        * weights,
        -1,
    )


def calculate_treelikelihood_discrete_cat(
    tip_partials: torch.Tensor,
    weights: torch.Tensor,
    post_indexing: torch.Tensor,
    mats: torch.Tensor,
    freqs: torch.Tensor,
    props: torch.Tensor,
) -> torch.Tensor:
    r"""Calculate log tree likelihood

    tip_partials is reshaped to [...,T,K,S,N] and concatenated with internal
    partials [...,I,K,S,N]

    :param tip_partials: list of tensors of partials [S,N]
    :param weights: [N]
    :param post_indexing: list of indexes in postorder
    :param mats: tensor of probability matrices [...,B,K,S,S]
    :param freqs: tensor of frequencies [...,1,S]
    :param props: tensor of proportions [...,K,1,1]
    :return: tree log likelihood [batch]
    """

    partials = torch.cat(
        (
            tip_partials.unsqueeze(1)
            .expand((-1, mats.shape[-5], -1, -1))
            .unsqueeze(-3),
            torch.empty(
                (tip_partials.shape[0] - 1,)
                + mats.shape[:-4]
                + (mats.shape[-3],)
                + tip_partials.shape[1:]
            ),
        ),
        0,
    )
    for indices in post_indexing:
        left = int(indices[1].item())
        right = int(indices[2].item())
        partials[indices[0]] = (mats[..., left, :, :, :] @ partials[left].clone()) * (
            mats[..., right, :, :, :] @ partials[right].clone()
        )
    return torch.sum(
        torch.log(freqs @ torch.sum(props * partials[post_indexing[-1][0]], -3))
        * weights,
        -1,
    )


def transform(preorder: torch.Tensor, bounds: torch.Tensor, x: torch.Tensor):
    heights = torch.empty_like(x)
    heights[..., -1] = x[..., -1]
    for i in range(preorder.shape[0]):
        parent_id = int(preorder[i][0])
        id_ = int(preorder[i][1])
        heights[..., id_] = bounds[id_] + x[..., id_] * (
            heights[..., parent_id] - bounds[id_]
        )
    return heights


def transform2(preorder: List[Tuple[int, int]], bounds: torch.Tensor, x: torch.Tensor):
    heights = torch.empty_like(x)
    heights[..., -1] = x[..., -1]
    for parent_id, id_ in preorder:
        heights[..., id_] = bounds[id_] + x[..., id_] * (
            heights[..., parent_id] - bounds[id_]
        )
    return heights


@torch.jit.script
def p_t(branch_lengths: torch.Tensor) -> torch.Tensor:
    """Calculate transition probability matrices.

    :param branch_lengths: tensor of branch lengths [B,K]
    :return: tensor of probability matrices [B,K,4,4]
    """
    d = torch.unsqueeze(branch_lengths, -1)
    a = 0.25 + 3.0 / 4.0 * torch.exp(-4.0 / 3.0 * d)
    b = 0.25 - 0.25 * torch.exp(-4.0 / 3.0 * d)
    return torch.cat((a, b, b, b, b, a, b, b, b, b, a, b, b, b, b, a), -1).reshape(
        d.shape[:-1] + (4, 4)
    )


def fluA_unrooted(args):
    tree, dna = read_tree_and_alignment(args.tree, args.input, True, True)
    branch_lengths = torch.tensor(
        [
            float(node.edge_length) * 0.001
            for node in sorted(
                list(tree.postorder_node_iter())[:-1], key=lambda x: x.index
            )
        ],
    )
    indices = []
    for node in tree.postorder_internal_node_iter():
        indices.append([node.index] + [child.index for child in node.child_nodes()])

    sequences = []
    taxa = []
    for taxon, seq in dna.items():
        sequences.append(Sequence(taxon.label, str(seq)))
        taxa.append(Taxon(taxon.label, None))

    partials, weights_tensor = compress_alignment(
        Alignment(None, sequences, Taxa(None, taxa), NucleotideDataType())
    )
    jc69_model = JC69('jc')
    freqs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    proportions = torch.tensor([[[1.0]]])

    print('treelikelihood v1')

    @benchmark
    def fn(bls):
        mats = jc69_model.p_t(bls)
        return calculate_treelikelihood_discrete(
            partials, weights_tensor, indices, mats, freqs, proportions
        )

    @benchmark
    def fn_grad(bls):
        mats = jc69_model.p_t(bls)
        log_prob = calculate_treelikelihood_discrete(
            partials, weights_tensor, indices, mats, freqs, proportions
        )
        log_prob.backward()
        return log_prob

    @benchmark
    def fn_rescaled(bls):
        mats = jc69_model.p_t(bls)
        return calculate_treelikelihood_discrete_rescaled(
            partials, weights_tensor, indices, mats, freqs, proportions
        )

    @benchmark
    def fn_grad_rescaled(bls):
        mats = jc69_model.p_t(bls)
        log_prob = calculate_treelikelihood_discrete_rescaled(
            partials, weights_tensor, indices, mats, freqs, proportions
        )
        log_prob.backward()
        return log_prob

    blens = branch_lengths.unsqueeze(0).unsqueeze(-1)

    total_time, log_prob = fn(args.replicates, blens)
    print(f'  {args.replicates} evaluations: {total_time} ({log_prob})')

    blens.requires_grad = True
    grad_total_time, log_prob = fn_grad(args.replicates, blens)
    print(f'  {args.replicates} gradient evaluations: {grad_total_time}')

    if torch.any(torch.isinf(log_prob)):
        blens.requires_grad = False
        total_time_r, log_prob_r = fn_rescaled(args.replicates, blens)
        print(
            f'  {args.replicates} evaluations rescaled: {total_time_r} ({log_prob_r})'
        )

        blens.requires_grad = True
        grad_total_time_r, log_prob_r = fn_grad_rescaled(args.replicates, blens)
        print(f'  {args.replicates} gradient evaluations rescaled: {grad_total_time_r}')

        if args.output:
            args.output.write(
                f"treelikelihood_rescaled,evaluation_,off,{total_time_r}\n"
            )
            args.output.write(
                f"treelikelihood_rescaled,gradient,off,{grad_total_time_r}\n"
            )

    if args.output:
        args.output.write(f"treelikelihood,evaluation,off,{total_time}\n")
        args.output.write(f"treelikelihood,gradient,off,{grad_total_time}\n")

    if args.all:
        tip_partials = torch.stack(partials[: len(sequences)])
        indices = torch.tensor(indices)

        print('treelikelihood v2 JIT off')

        @benchmark
        def fn2(bls):
            mats = p_t(bls)
            return calculate_treelikelihood_discrete_split(
                tip_partials, weights_tensor, indices, mats, freqs, proportions
            )

        @benchmark
        def fn2_grad(bls):
            mats = p_t(bls)
            log_prob = calculate_treelikelihood_discrete_split(
                tip_partials, weights_tensor, indices, mats, freqs, proportions
            )
            log_prob.backward()
            return log_prob

        with torch.no_grad():
            total_time, log_prob = fn2(args.replicates, blens)
        print(f'  {args.replicates} evaluations: {total_time} ({log_prob})')

        total_time, log_prob = fn2_grad(args.replicates, blens)
        print(f'  {args.replicates} gradient evaluations: {total_time}')

        print('treelikelihood v2 JIT on')
        like2_jit = torch.jit.script(calculate_treelikelihood_discrete_split)

        @benchmark
        def fn2_jit(bls):
            mats = p_t(bls)
            return like2_jit(
                tip_partials, weights_tensor, indices, mats, freqs, proportions
            )

        @benchmark
        def fn2_grad_jit(bls):
            mats = p_t(bls)
            log_prob = like2_jit(
                tip_partials, weights_tensor, indices, mats, freqs, proportions
            )
            log_prob.backward()
            return log_prob

        with torch.no_grad():
            total_time, log_prob = fn2_jit(args.replicates, blens)
        print(f'  {args.replicates} evaluations: {total_time}')

        total_time, log_prob = fn2_grad_jit(args.replicates, blens)
        print(f'  {args.replicates} gradient evaluations: {total_time}')

        print('treelikelihood v3 JIT off')

        @benchmark
        def fn3(bls):
            mats = p_t(bls)
            return calculate_treelikelihood_discrete_cat(
                tip_partials, weights_tensor, indices, mats, freqs, proportions
            )

        @benchmark
        def fn3_grad(bls):
            mats = p_t(bls)
            log_prob = calculate_treelikelihood_discrete_cat(
                tip_partials, weights_tensor, indices, mats, freqs, proportions
            )
            log_prob.backward()
            return log_prob

        with torch.no_grad():
            total_time, log_prob = fn3(args.replicates, blens)
        print(f'  {args.replicates} evaluations: {total_time} ({log_prob})')

        total_time, log_prob = fn3_grad(args.replicates, blens)
        print(f'  {args.replicates} gradient evaluations: {total_time}')

        print('treelikelihood v3 JIT on')
        like3_jit = torch.jit.script(calculate_treelikelihood_discrete_cat)

        @benchmark
        def fn3_jit(bls):
            mats = p_t(bls)
            return like3_jit(
                tip_partials, weights_tensor, indices, mats, freqs, proportions
            )

        @benchmark
        def fn3_grad_jit(bls):
            mats = p_t(bls)
            log_prob = like3_jit(
                tip_partials, weights_tensor, indices, mats, freqs, proportions
            )
            log_prob.backward()
            return log_prob

        with torch.no_grad():
            total_time, log_prob = fn3_jit(args.replicates, blens)
        print(f'  {args.replicates} evaluations: {total_time} ({log_prob})')

        total_time, log_prob = fn3_grad_jit(args.replicates, blens)
        print(f'  {args.replicates} gradient evaluations: {total_time}')


def ratio_transform_jacobian(args):
    tree = read_tree(args.tree, True, True)
    taxa = []
    for node in tree.leaf_node_iter():
        taxa.append(Taxon(node.label, {'date': node.date}))
    taxa_count = len(taxa)
    ratios_root_height = Parameter(
        "internal_heights", torch.tensor([0.5] * (taxa_count - 1) + [20])
    )
    tree_model = ReparameterizedTimeTreeModel(
        "tree", tree, Taxa('taxa', taxa), ratios_root_height
    )

    ratios_root_height.tensor = tree_model.transform.inv(
        heights_from_branch_lengths(tree)
    )

    @benchmark
    def fn(ratios_root_height):
        internal_heights = tree_model.transform(ratios_root_height)
        return tree_model.transform.log_abs_det_jacobian(
            ratios_root_height, internal_heights
        )

    @benchmark
    def fn_grad(ratios_root_height):
        internal_heights = tree_model.transform(ratios_root_height)
        log_det_jac = tree_model.transform.log_abs_det_jacobian(
            ratios_root_height, internal_heights
        )
        log_det_jac.backward()
        return log_det_jac

    print('  JIT off')
    total_time, log_det_jac = fn(args.replicates, ratios_root_height.tensor)
    print(f'  {args.replicates} evaluations: {total_time} ({log_det_jac})')

    ratios_root_height.requires_grad = True
    grad_total_time, log_det_jac = fn_grad(args.replicates, ratios_root_height.tensor)
    print(
        f'  {args.replicates} gradient evaluations: {grad_total_time} ({log_det_jac})'
    )

    if args.output:
        args.output.write(f"ratio_transform_jacobian,evaluation,off,{total_time}\n")
        args.output.write(f"ratio_transform_jacobian,gradient,off,{grad_total_time}\n")

    if args.debug:
        internal_heights = tree_model.transform(ratios_root_height.tensor)
        log_det_jac = tree_model.transform.log_abs_det_jacobian(
            ratios_root_height.tensor, internal_heights
        )
        log_det_jac.backward()
        print(ratios_root_height.grad)


def ratio_transform(args):
    replicates = args.replicates
    tree = read_tree(args.tree, True, True)
    taxa_count = len(tree.taxon_namespace)
    taxa = []
    for node in tree.leaf_node_iter():
        taxa.append(Taxon(node.label, {'date': node.date}))
    ratios_root_height = Parameter(
        "internal_heights", torch.tensor([0.5] * (taxa_count - 2) + [10])
    )
    tree_model = ReparameterizedTimeTreeModel(
        "tree", tree, Taxa('taxa', taxa), ratios_root_height
    )

    ratios_root_height.tensor = tree_model.transform.inv(
        heights_from_branch_lengths(tree)
    )

    @benchmark
    def fn(ratios_root_height):
        return tree_model.transform(
            ratios_root_height,
        )

    @benchmark
    def fn_grad(ratios_root_height):
        heights = tree_model.transform(
            ratios_root_height,
        )
        heights.backward(torch.ones_like(ratios_root_height))
        ratios_root_height.grad.data.zero_()
        return heights

    total_time, heights = fn(args.replicates, ratios_root_height.tensor)
    print(f'  {replicates} evaluations: {total_time}')

    ratios_root_height.requires_grad = True
    grad_total_time, heights = fn_grad(args.replicates, ratios_root_height.tensor)
    print(f'  {replicates} gradient evaluations: {grad_total_time}')

    if args.output:
        args.output.write(f"ratio_transform_jacobian,evaluation,off,{total_time}\n")
        args.output.write(f"ratio_transform_jacobian,gradient,off,{grad_total_time}\n")

    print('  JIT off')

    @benchmark
    def fn2(ratios_root_height):
        return transform(
            tree_model.transform._forward_indices,
            tree_model.transform._bounds,
            ratios_root_height,
        )

    @benchmark
    def fn2_grad(ratios_root_height):
        heights = transform(
            tree_model.transform._forward_indices,
            tree_model.transform._bounds,
            ratios_root_height,
        )
        heights.backward(torch.ones_like(ratios_root_height))
        ratios_root_height.grad.data.zero_()
        return heights

    ratios_root_height.requires_grad = False
    total_time, heights = fn2(args.replicates, ratios_root_height.tensor)
    print(f'  {replicates} evaluations: {total_time}')

    ratios_root_height.requires_grad = True
    total_time, heights = fn2_grad(args.replicates, ratios_root_height.tensor)
    print(f'  {replicates} gradient evaluations: {total_time}')

    print('  JIT on')
    transform_script = torch.jit.script(transform)

    @benchmark
    def fn2_jit(ratios_root_height):
        return transform_script(
            tree_model.transform._forward_indices,
            tree_model.transform._bounds,
            ratios_root_height,
        )

    @benchmark
    def fn2_grad_jit(ratios_root_height):
        heights = transform_script(
            tree_model.transform._forward_indices,
            tree_model.transform._bounds,
            ratios_root_height,
        )
        heights.backward(torch.ones_like(ratios_root_height))
        ratios_root_height.grad.data.zero_()
        return heights

    ratios_root_height.requires_grad = False
    total_time, heights = fn2_jit(args.replicates, ratios_root_height.tensor)
    print(f'  {replicates} evaluations: {total_time}')

    ratios_root_height.requires_grad = True
    total_time, heights = fn2_grad_jit(args.replicates, ratios_root_height.tensor)
    print(f'  {replicates} gradient evaluations: {total_time}')

    print('ratio_transform v2 JIT off')

    @benchmark
    def fn3(ratios_root_height):
        return transform2(
            tree_model.transform._forward_indices.tolist(),
            tree_model.transform._bounds,
            ratios_root_height,
        )

    @benchmark
    def fn3_grad(ratios_root_height):
        heights = transform2(
            tree_model.transform._forward_indices.tolist(),
            tree_model.transform._bounds,
            ratios_root_height,
        )
        heights.backward(torch.ones_like(ratios_root_height))
        ratios_root_height.grad.data.zero_()
        return heights

    ratios_root_height.requires_grad = False
    total_time, heights = fn3(args.replicates, ratios_root_height.tensor)
    print(f'  {replicates} evaluations: {total_time}')

    ratios_root_height.requires_grad = True
    total_time, heights = fn3_grad(args.replicates, ratios_root_height.tensor)
    print(f'  {replicates} gradient evaluations: {total_time}')

    print('ratio_transform v2 JIT on')
    transform2_script = torch.jit.script(transform2)

    @benchmark
    def fn3_jit(ratios_root_height):
        return transform2_script(
            tree_model.transform._forward_indices.tolist(),
            tree_model.transform._bounds,
            ratios_root_height,
        )

    @benchmark
    def fn3_grad_jit(ratios_root_height):
        heights = transform2_script(
            tree_model.transform._forward_indices.tolist(),
            tree_model.transform._bounds,
            ratios_root_height,
        )
        heights.backward(torch.ones_like(ratios_root_height))
        ratios_root_height.grad.data.zero_()
        return heights

    ratios_root_height.requires_grad = False
    total_time, heights = fn3_jit(args.replicates, ratios_root_height.tensor)
    print(f'  {replicates} evaluations: {total_time}')

    ratios_root_height.requires_grad = True
    total_time, heights = fn3_grad_jit(args.replicates, ratios_root_height.tensor)
    print(f'  {replicates} gradient evaluations: {total_time}')


def constant_coalescent(args):
    tree = read_tree(args.tree, True, True)
    taxa_count = len(tree.taxon_namespace)
    taxa = []
    for node in tree.leaf_node_iter():
        taxa.append(Taxon(node.label, {'date': node.date}))
    ratios_root_height = Parameter(
        "internal_heights", torch.tensor([0.5] * (taxa_count - 2) + [20.0])
    )
    tree_model = ReparameterizedTimeTreeModel(
        "tree", tree, Taxa('taxa', taxa), ratios_root_height
    )

    ratios_root_height.tensor = tree_model.transform.inv(
        heights_from_branch_lengths(tree)
    )
    pop_size = torch.tensor([4.0])

    print('JIT off')

    @benchmark
    def fn(tree_model, pop_size):
        tree_model.heights_need_update = True
        return ConstantCoalescent(pop_size).log_prob(tree_model.node_heights)

    @benchmark
    def fn_grad(tree_model, pop_size):
        tree_model.heights_need_update = True
        log_p = ConstantCoalescent(pop_size).log_prob(tree_model.node_heights)
        log_p.backward()
        ratios_root_height.tensor.grad.data.zero_()
        pop_size.grad.data.zero_()
        return log_p

    total_time, log_p = fn(args.replicates, tree_model, pop_size)
    print(f'  {args.replicates} evaluations: {total_time} {log_p}')

    ratios_root_height.requires_grad = True
    pop_size.requires_grad_(True)
    grad_total_time, log_p = fn(args.replicates, tree_model, pop_size)
    print(f'  {args.replicates} gradient evaluations: {grad_total_time}')

    if args.output:
        args.output.write(f"coalescent,evaluation,off,{total_time}\n")
        args.output.write(f"coalescent,gradient,off,{grad_total_time}\n")

    if args.debug:
        tree_model.heights_need_update = True
        log_p = ConstantCoalescent(pop_size).log_prob(tree_model.node_heights)
        log_p.backward()
        print('gradient ratios: ', ratios_root_height.grad)
        print('gradient pop size: ', pop_size.grad)
        ratios_root_height.tensor.grad.data.zero_()
        pop_size.grad.data.zero_()

    print('JIT on')
    log_prob_script = torch.jit.script(log_prob)

    @benchmark
    def fn_jit(tree_model, pop_size):
        tree_model.heights_need_update = True
        return log_prob_script(tree_model.node_heights, pop_size)

    @benchmark
    def fn_grad_jit(tree_model, pop_size):
        tree_model.heights_need_update = True
        log_p = log_prob_script(tree_model.node_heights, pop_size)
        log_p.backward()
        ratios_root_height.tensor.grad.data.zero_()
        pop_size.grad.data.zero_()
        return log_p

    ratios_root_height.requires_grad = False
    pop_size.requires_grad_(False)
    total_time, log_p = fn_jit(args.replicates, tree_model, pop_size)
    print(f'  {args.replicates} evaluations: {total_time} {log_p}')

    ratios_root_height.requires_grad = True
    pop_size.requires_grad_(True)
    grad_total_time, log_p = fn_grad_jit(args.replicates, tree_model, pop_size)
    print(f'  {args.replicates} gradient evaluations: {grad_total_time}')

    if args.output:
        args.output.write(f"coalescent,evaluation,on,{total_time}\n")
        args.output.write(f"coalescent,gradient,on,{grad_total_time}\n")

    if args.all:
        print('make sampling times unique and count them:')

        @benchmark
        def fn3(tree_model, pop_size):
            tree_model.heights_need_update = True
            node_heights = torch.cat(
                (x, tree_model.node_heights[..., tree_model.taxa_count :])
            )
            return log_prob_squashed(
                pop_size, node_heights, counts, tree_model.taxa_count
            )

        @benchmark
        def fn3_grad(tree_model, ratios_root_height, pop_size):
            tree_model.heights_need_update = True
            node_heights = torch.cat(
                (x, tree_model.node_heights[..., tree_model.taxa_count :])
            )
            log_p = log_prob_squashed(
                pop_size, node_heights, counts, tree_model.taxa_count
            )
            log_p.backward()
            ratios_root_height.tensor.grad.data.zero_()
            pop_size.grad.data.zero_()
            return log_p

        x, counts = torch.unique(tree_model.sampling_times, return_counts=True)
        counts = torch.cat((counts, torch.tensor([-1] * (taxa_count - 1))))

        with torch.no_grad():
            total_time, log_p = fn3(args.replicates, tree_model, pop_size)
        print(f'  {args.replicates} evaluations: {total_time} ({log_p})')

        total_time, log_p = fn3_grad(
            args.replicates, tree_model, ratios_root_height, pop_size
        )
        print(f'  {args.replicates} gradient evaluations: {total_time}')


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help="""Alignment file""")
parser.add_argument('-t', '--tree', required=True, help="""Tree file""")
parser.add_argument(
    '--replicates',
    required=True,
    type=int,
    help="""Number of replicates""",
)
parser.add_argument(
    "-o",
    "--output",
    type=argparse.FileType("w"),
    default=None,
    help="""csv output file""",
)
parser.add_argument(
    "-d",
    "--dtype",
    choices=('float32', 'float64'),
    default='float64',
    help="""double or single precision""",
)
parser.add_argument(
    '--debug', required=False, action='store_true', help="""Debug mode"""
)
parser.add_argument("--all", required=False, action="store_true", help="""Run all""")
args = parser.parse_args()

if args.dtype == 'float64':
    torch.set_default_dtype(torch.float64)
else:
    torch.set_default_dtype(torch.float32)

if args.output:
    args.output.write("function,mode,JIT,time\n")

print('Tree likelihood unrooted:')
fluA_unrooted(args)
print()

print('Height transform log det Jacobian:')
ratio_transform_jacobian(args)
print()

print('Node height transform:')
ratio_transform(args)

print()
print('Constant coalescent:')
constant_coalescent(args)

if args.output:
    args.output.close()
