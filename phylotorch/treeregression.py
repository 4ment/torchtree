import numpy as np


def regression(tree):
    dic = {}
    taxa_count = len(tree.taxon_namespace)
    bls = np.empty(taxa_count, dtype=np.float)
    ts = np.empty(taxa_count, dtype=np.float)

    for node in tree.postorder_node_iter():
        if node.is_leaf():
            dic[node] = (node,)
            bls[node.index] = node.edge_length
            ts[node.index] = node.original_date
        elif node != tree.seed_node:
            children = node.child_nodes()
            dic[node] = dic[children[0]] + dic[children[1]]
            for c in dic[node]:
                bls[c.index] += node.edge_length

    sumX = np.sum(ts)
    sumY = np.sum(bls)
    sumXX = np.sum(np.power(ts, 2.0))
    sumXY = np.sum(ts * bls)
    sumSqDevX = sumXX - np.power(sumX, 2.0) / taxa_count
    sumSqDevXY = sumXY - sumX * sumY / taxa_count

    slope = sumSqDevXY / sumSqDevX
    intercept = (sumY - slope * sumX) / taxa_count
    return slope, -intercept / slope
