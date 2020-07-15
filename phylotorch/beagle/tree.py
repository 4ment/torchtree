import numpy as np
import torch
import torch.autograd


class NodeHeightTransform(object):
    def __init__(self, inst):
        self.inst = inst

    def __call__(self, ratios, root_height):
        transform = NodeHeightAutogradFunction.apply
        return transform(torch.cat((ratios, root_height)))


class NodeHeightAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inst, ratios_root_height):
        ctx.inst = inst
        node_heights = torch.tensor(
            np.array(inst.tree_collection.trees[0].node_heights, copy=True)[ratios_root_height.shape[0] + 1:])
        return node_heights

    @staticmethod
    def backward(ctx, grad_output):
        return None, torch.tensor(
            ctx.inst.tree_collection.trees[0].ratio_gradient_of_input_gradient(grad_output.numpy()))


def get_inorder_from_libsbn(parent_id_vector):
    stack = []
    node_id = parent_id_vector.shape[0]
    children = np.nonzero(parent_id_vector == node_id)[0]
    indexes = []
    stack.extend((children[0], children[1]))
    indexes.append((node_id, children[0]))
    indexes.append((node_id, children[1]))
    while len(stack) > 0:
        node_id = stack.pop()
        children = np.nonzero(parent_id_vector == node_id)[0]
        if len(children) > 0:
            stack.extend((children[0], children[1]))
            indexes.append((node_id, children[0]))
            indexes.append((node_id, children[1]))
    return indexes
