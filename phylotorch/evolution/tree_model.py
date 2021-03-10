import abc

import numpy as np
import torch
from dendropy import Tree, TaxonNamespace
from torch.distributions.transforms import Transform

from ..core.model import Model, Parameter
from ..core.utils import process_object


class BranchLengthTransform(object):
    def __init__(self, bounds, indexing):
        self.bounds = bounds
        self.indexing = indexing

    def __call__(self, node_heights):
        return heights_to_branch_lengths(node_heights, self.bounds, self.indexing)


class NodeHeightTransform(Transform):
    r"""
        Transform from ratios to node heights.
    """
    bijective = True
    sign = +1

    def __init__(self, bounds, indexing, cache_size=0):
        super(NodeHeightTransform, self).__init__(cache_size=cache_size)
        self.bounds = bounds
        self.indexing = indexing
        self.taxa_count = int((bounds.shape[0] + 1) / 2)
        self.indices_sorted = indexing[np.argsort(indexing[:, 1])].transpose()[0, self.taxa_count:] - self.taxa_count

    def _call(self, x):
        return transform_ratios(x, self.bounds, self.indexing)

    def _inverse(self, y):
        raise NotImplementedError

    def log_abs_det_jacobian(self, x, y):
        # return torch.log(
        # y[self.indices_sorted[0, self.taxa_count:] - self.taxa_count] - self.bounds[
        #                                                                 self.taxa_count:-1]).sum()
        return torch.log(
            y[self.indices_sorted] - self.bounds[self.taxa_count:-1]).sum()


class GeneralNodeHeightTransform(Transform):
    r"""
        Transform from ratios to node heights.
    """
    bijective = True
    sign = +1

    def __init__(self, tree, cache_size=0):
        super(GeneralNodeHeightTransform, self).__init__(cache_size=cache_size)
        self.tree = tree
        self.taxa_count = int((self.tree.bounds.shape[0] + 1) / 2)
        self.indices_sorted = self.tree.preorder[np.argsort(self.tree.preorder[:, 1])].transpose()[0,
                              self.taxa_count:] - self.taxa_count

    def _call(self, x):
        return transform_ratios(x, self.tree.bounds, self.tree.preorder)

    def _inverse(self, y):
        raise NotImplementedError

    def log_abs_det_jacobian(self, x, y):
        # return torch.log(
        # y[self.indices_sorted[0, self.taxa_count:] - self.taxa_count] - self.bounds[
        #                                                                 self.taxa_count:-1]).sum()
        return torch.log(
            y[self.indices_sorted] - self.tree.bounds[self.taxa_count:-1]).sum()


def heights_to_branch_lengths(node_heights, bounds, indexing):
    taxa_count = int((bounds.shape[0] + 1) / 2)
    indices_sorted = indexing[np.argsort(indexing[:, 1])].transpose()
    return torch.cat((node_heights[indices_sorted[0, :taxa_count] - taxa_count] - bounds[:taxa_count],
                      node_heights[indices_sorted[0, taxa_count:] - taxa_count] - node_heights[
                          indices_sorted[1, taxa_count:] - taxa_count]))


def transform_ratios(ratios_root_height, bounds, indexing):
    ### # type: (Tensor, Tensor, Tuple[Tuple[int, int], ...]) -> Tensor
    """
    Transform node ratios/root height to internal node heights.
    :param root_height: root height
    :param ratios: node ratios of internal nodes
    :param bounds: lower bound of each node
    :param indexing: pairs of parent/child indices, must be preorder or inorder
    :return: internal node heights
    """
    taxa_count = ratios_root_height.shape[0] + 1
    heights = ratios_root_height.clone()
    for parent_id, id_ in indexing:
        if id_ >= taxa_count:
            heights[id_ - taxa_count] = bounds[id_] + ratios_root_height[id_ - taxa_count] * (
                    heights[parent_id - taxa_count] - bounds[id_])
    return heights


def setup_indexes(tree):
    for node in tree.postorder_node_iter():
        node.index = -1
        node.annotations.add_bound_attribute("index")

    s = len(tree.taxon_namespace)
    taxa_dict = {taxon.label: idx for idx, taxon in enumerate(tree.taxon_namespace)}

    for node in tree.postorder_node_iter():
        if not node.is_leaf():
            node.index = s
            s += 1
        else:
            node.index = taxa_dict[node.taxon.label]


def setup_dates(tree, heterochronous=False):
    # parse dates
    if heterochronous:
        dates = {}
        for node in tree.leaf_node_iter():
            dates[str(node.taxon)] = float(str(node.taxon).split('_')[-1][:-1])

        max_date = max(dates.values())
        min_date = min(dates.values())
        print(min_date, max_date)

        # time starts at 0
        if min_date == 0:
            for node in tree.leaf_node_iter():
                node.date = dates[str(node.taxon)]
                node.original_date = dates[str(node.taxon)]
            oldest = max_date
        # time is a year
        else:
            for node in tree.leaf_node_iter():
                node.date = max_date - dates[str(node.taxon)]
                node.original_date = dates[str(node.taxon)]
            oldest = max_date - min_date
    else:
        for node in tree.postorder_node_iter():
            node.date = 0.0
            node.original_date = 0.0
        oldest = None

    return oldest


def initialize_dates_from_taxa(tree, taxa):
    dates = [taxon['date'] for taxon in taxa]
    max_date = max(dates)

    # parse dates
    if max_date != 0.0:
        # time starts at 0
        if min(dates) == 0.0:
            for node in tree.leaf_node_iter():
                node.date = taxa[node.index]['date']
                node.original_date = node.date
        # time is a year
        else:
            for node in tree.leaf_node_iter():
                node.date = max_date - taxa[node.index]['date']
                node.original_date = taxa[node.index]['date']
    else:
        for node in tree.leaf_node_iter():
            node.date = 0.0
            node.original_date = 0.0


def heights_from_branch_lengths(tree):
    heights = np.empty(2 * len(tree.taxon_namespace) - 1)
    for node in tree.postorder_node_iter():
        if node.is_leaf():
            heights[node.index] = node.date
        else:
            child = next(node.child_node_iter())
            heights[node.index] = heights[child.index] + float(child.edge_length)
    return heights[len(tree.taxon_namespace):]


class Node:
    def __init__(self, name, height=0.0):
        self.name = name
        self.height = height
        self.parent = None
        self.children = []

    def __iter__(self):
        if len(self.children) > 0:
            for c in self.children[0]:
                yield c
            for c in self.children[1]:
                yield c
        yield self


def random_tree_from_heights(sampling, heights):
    nodes = [Node('taxon{}'.format(idx), height=s) for idx, s in enumerate(sampling)]

    for i, height in enumerate(heights):
        indexes = []
        for idx, node in enumerate(nodes):
            if node.height < height:
                indexes.append(idx)
        idx1 = idx2 = np.random.randint(0, len(indexes))
        while idx1 == idx2:
            idx2 = np.random.randint(0, len(indexes))
        new_node = Node('node{}'.format(len(nodes)), height=height)
        idx1, idx2 = sorted([idx1, idx2])
        new_node.children = (nodes[indexes[idx1]], nodes[indexes[idx2]])
        nodes[idx1].parent = nodes[idx2].parent = new_node
        nodes[idx1] = new_node
        del nodes[idx2]
    return nodes[0]


class TreeModel(Model):
    _tag = 'tree_model'

    @classmethod
    def from_json(cls, data, dic):
        pass

    def handle_model_changed(self, model, obj, index):
        pass

    def handle_parameter_changed(self, variable, index, event):
        pass

    def __init__(self, id_, branches, postorder, preorder=None, bounds=None):
        self._branches = branches
        self.postorder = postorder
        self.preorder = preorder
        self.bounds = bounds
        super().__init__(id_)

    def branch_lengths(self):
        return self._branches.tensor  # if self.bounds is not None else torch.cat(
        # (self.branches, torch.zeros(1, dtype=self.branches.dtype)))

    def update(self, value):
        if isinstance(value, dict):
            if self._branches.id in value:
                self._branches.tensor = value[self._branches.id]
        else:
            self.branches = value


def parse_tree(taxa, data):
    taxon_namespace = TaxonNamespace([taxon.id for taxon in taxa])
    if 'newick' in data:
        tree = Tree.get(data=data['newick'], schema='newick', preserve_underscores=True,
                        rooting='force-rooted', taxon_namespace=taxon_namespace)
    elif 'file' in data:
        tree = Tree.get(path=data['file'], schema='newick', preserve_underscores=True,
                        rooting='force-rooted', taxon_namespace=taxon_namespace)
    else:
        raise ValueError('Tree model requires a file or newick element to be specified')
    tree.resolve_polytomies(update_bipartitions=True)
    setup_indexes(tree)
    return tree


class AbstractTreeModel(Model):
    def __init__(self, id_, tree):
        self.tree = tree
        self.taxa_count = len(tree.taxon_namespace)
        self.postorder = []
        self.preorder = []
        self.update_traversals()
        super(AbstractTreeModel, self).__init__(id_)

    def update_traversals(self):
        # postorder for peeling
        self.postorder = []
        for node in self.tree.postorder_node_iter():
            if not node.is_leaf():
                children = node.child_nodes()
                self.postorder.append((node.index, children[0].index, children[1].index))

        # preoder indexing to go from ratios to heights
        self.preorder = np.array(
            [(node.parent_node.index, node.index) for node in self.tree.preorder_node_iter() if
             node != self.tree.seed_node])

    def handle_model_changed(self, model, obj, index):
        pass

    @abc.abstractmethod
    def branch_lengths(self):
        pass


class UnRootedTreeModel(AbstractTreeModel):
    def __init__(self, id_, tree, branch_lengths):
        super(UnRootedTreeModel, self).__init__(id_, tree)
        self._branch_lengths = branch_lengths
        self.add_parameter(branch_lengths)

    def branch_lengths(self):
        return self._branch_lengths.tensor  # if self.bounds is not None else torch.cat(
        # (self.branches, torch.zeros(1, dtype=self.branches.dtype)))

    def update(self, value):
        if isinstance(value, dict):
            if self._branch_lengths.id in value:
                self._branch_lengths.tensor = value[self._branch_lengths.id]
        else:
            self.branches = value

    def handle_parameter_changed(self, variable, index, event):
        self.fire_model_changed()

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        taxa = process_object(data['taxa'], dic)
        tree = parse_tree(taxa, data)

        taxa_count = len(tree.taxon_namespace)
        if isinstance(data['branch_lengths'], dict) and 'tensor' not in data['branch_lengths']:
            branch_lengths_id = data['branch_lengths']['id']
            bls = torch.tensor(np.array(
                [float(node.edge_length) for node in
                 sorted(list(tree.postorder_node_iter())[:-1], key=lambda x: x.index)]))
            branch_lengths = Parameter(branch_lengths_id, bls)
            dic[branch_lengths_id] = branch_lengths
        else:
            branch_lengths = process_object(data['branch_lengths'], dic)

        return cls(id_, tree, branch_lengths)


class TimeTreeModel(AbstractTreeModel):

    def __init__(self, id_, tree, node_heights):
        super(TimeTreeModel, self).__init__(id_, tree)
        self._node_heights = node_heights
        self.taxa_count = len(tree.taxon_namespace)
        self.bounds = self.create_bounds(tree)
        self.sampling_times = self.bounds[:self.taxa_count]
        if node_heights is not None:
            self.add_parameter(node_heights)

    def create_bounds(self, tree):
        bounds = np.empty(2 * len(tree.taxon_namespace) - 1)
        for node in tree.postorder_node_iter():
            if node.is_leaf():
                bounds[node.index] = node.date
            else:
                bounds[node.index] = np.max([bounds[x.index] for x in node.child_node_iter()])
        return torch.tensor(bounds)

    @property
    def node_heights(self):
        return self._node_heights.tensor

    def branch_lengths(self):
        return heights_to_branch_lengths(self._node_heights.tensor, self.bounds, self.preorder)

    def update(self, value):
        if isinstance(value, dict):
            if self._branches.id in value:
                self._branches.tensor = value[self._branches.id]
        else:
            self.branches = value

    def handle_parameter_changed(self, variable, index, event):
        self.fire_model_changed()

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        taxa = process_object(data['taxa'], dic)
        tree = parse_tree(taxa, data)
        initialize_dates_from_taxa(tree, taxa)

        if isinstance(data['node_heights'], dict) and len(data['node_heights']) == 1:
            node_heights_id = data['node_heights']['id']
            heights_np = heights_from_branch_lengths(tree)
            node_heights = Parameter(node_heights_id, torch.tensor(heights_np))
            dic[node_heights_id] = node_heights
            tree_model = cls(id_, tree, node_heights)
        else:
            tree_model = cls(id_, tree, None)
            dic[id_] = tree_model
            tree_model._node_heights = process_object(data['node_heights'], dic)
            tree_model.add_parameter(tree_model._node_heights)

        return tree_model
