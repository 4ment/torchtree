from __future__ import annotations

import torch
from treezy import Tree


def linear_regression(tree: Tree) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate rate and root height using linear regression.

    :param tree: Treezy tree
    :returns:
        - rate - substitution rate
        - root_height - root height
    """
    dic = {}
    taxa_count = len(tree.taxon_names)
    bls = [0.0] * taxa_count
    ts = [0.0] * taxa_count

    for node in tree.postorder():
        if node.is_leaf:
            dic[node] = (node,)
            bls[node.index] = node.distance
            ts[node.index] = node.original_date
        elif not node.is_root:
            children = node.children
            dic[node] = dic[children[0]] + dic[children[1]]
            for c in dic[node]:
                bls[c.index] += node.distance
    ts = torch.tensor(ts)
    bls = torch.tensor(bls)
    sumX = torch.sum(ts)
    sumY = torch.sum(bls)
    sumXX = torch.sum(ts**2.0)
    sumXY = torch.sum(ts * bls)
    sumSqDevX = sumXX - torch.pow(sumX, 2.0) / taxa_count
    sumSqDevXY = sumXY - sumX * sumY / taxa_count

    slope = sumSqDevXY / sumSqDevX
    intercept = (sumY - slope * sumX) / taxa_count
    root_height = -intercept / slope
    return slope, root_height
