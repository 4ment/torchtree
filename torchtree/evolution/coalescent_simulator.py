import dendropy
import torch
from torch import Tensor
from torch.distributions import Exponential


def sample_coalescent_times(
    sampling_times: Tensor, sampling_counts: Tensor, trajectory, lower_bound: float
):
    r"""Simulate from inhomogeneous, heterochronous coalescent using thinning algorithm.

    # Code adapted from https://github.com/mdkarcher/phylodyn/R/coalsieve.R

    :param Tensor sampling_times: one-dimentional tensor containing sampling times.
    :param Tensor sampling_counts: samples taken per sampling time.
    :param trajectory: function that returns effective population size at time t.
    :param float lower_bound: lower limit of trajectory function on its support.
    :return: coalescent times.
    :rtype: Tensor

    :example:
    >>> unif_traj = lambda t: 1.0
    >>> times = sample_coalescent_times(torch.arange(3.0), torch.tensor([3,2,1]), unif_traj, lower_bound=0.1)
    >>> times.shape
    torch.Size([5])
    >>> torch.all(times > 0)
    tensor(True)
    >>> torch.all(times[:-1] <= times[1:])  # non-decreasing
    tensor(True)
    """
    coalescent_times = []
    sampling_counts = sampling_counts.tolist()

    sampling_index = 0
    active_lineages = sampling_counts[sampling_index]
    time = sampling_times[sampling_index]
    max_sampling_times = max(sampling_times)

    while time <= max_sampling_times or active_lineages > 1:
        if active_lineages == 1:
            sampling_index += 1
            active_lineages += sampling_counts[sampling_index]
            time = sampling_times[sampling_index]

        rate = 0.5 * active_lineages * (active_lineages - 1) / lower_bound
        time = time + Exponential(rate).sample().item()

        if (
            sampling_index < len(sampling_times) - 1
            and time >= sampling_times[sampling_index + 1]
        ):
            sampling_index += 1
            active_lineages += sampling_counts[sampling_index]
            time = sampling_times[sampling_index]
        elif torch.rand(1) <= lower_bound / trajectory(time):
            coalescent_times.append(time)
            active_lineages -= 1

    return torch.tensor(coalescent_times)


def sample_tree(
    sampling_times: Tensor, sampling_counts: Tensor, coalescent_times: Tensor
):
    r"""Generate a tree from coalescent times, sampling times, and sampling counts.

    :param Tensor sampling_times: sampling times.
    :param Tensor sampling_counts: samples taken per sampling time.
    :param Tensor coalescent_times: coalescent times.
    :return: A dendropy tree object.
    """
    taxon_count = sum(sampling_counts).item()
    taxon_namespace = dendropy.TaxonNamespace([f"t{i}" for i in range(taxon_count)])
    active_nodes = []
    active_taxon_count = sampling_counts[0].item()  # used for naming taxa t0, t1, ...

    for i in range(active_taxon_count):
        node = dendropy.Node(edge_length=0.0)
        node.taxon = taxon_namespace.get_taxon(f"t{i}")
        node.date = sampling_times[0].item()
        active_nodes.append(node)

    node_heights = torch.cat((sampling_times, coalescent_times))
    events = torch.cat(
        (
            torch.full(sampling_times.shape, 0),
            torch.full(coalescent_times.shape, 1),
        )
    )
    indices = torch.argsort(node_heights, descending=False)
    node_heights = torch.gather(node_heights, -1, indices)
    events = torch.gather(events, -1, indices)

    sampling_index = 1
    for j in range(1, len(events)):
        if events[j] == 1:
            # coalescent
            assert len(active_nodes) >= 2
            idx0, idx1 = torch.multinomial(torch.ones(len(active_nodes)), 2).sort()[0]
            node = dendropy.Node(edge_length=node_heights[j])
            active_nodes[idx0].edge_length = (
                node_heights[j] - active_nodes[idx0].edge_length
            )
            active_nodes[idx1].edge_length = (
                node_heights[j] - active_nodes[idx1].edge_length
            )
            node.add_child(active_nodes[idx0])
            node.add_child(active_nodes[idx1])
            active_nodes[idx0] = node
            active_nodes.pop(idx1)
        else:
            # sampling
            for i in range(sampling_counts[sampling_index]):
                node = dendropy.Node(edge_length=node_heights[j])
                node.taxon = taxon_namespace.get_taxon(f"t{i+active_taxon_count}")
                node.date = sampling_times[sampling_index].item()
                active_nodes.append(node)
            active_taxon_count += sampling_counts[sampling_index]
            sampling_index += 1

    assert len(active_nodes) == 1
    active_nodes[0].edge_length = None
    tree = dendropy.Tree(seed_node=active_nodes[0], taxon_namespace=taxon_namespace)
    return tree
