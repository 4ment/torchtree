import abc
from abc import ABC
from io import StringIO
from typing import List, Optional, Union

import numpy as np
import torch
from dendropy import TaxonNamespace, Tree
from torch.distributions.transforms import Transform

from ..core.model import Model, Parameter
from ..core.utils import process_object
from ..typing import ID
from .taxa import Taxa


class GeneralNodeHeightTransform(Transform):
    r"""
    Transform from ratios to node heights.
    """
    bijective = True
    sign = +1

    def __init__(self, tree: 'TimeTreeModel', cache_size=0) -> None:
        super().__init__(cache_size=cache_size)
        self.tree = tree
        self.taxa_count = self.tree.taxa_count
        self.indices_sorted = (
            self.tree.preorder[np.argsort(self.tree.preorder[:, 1])].transpose()[
                0, self.taxa_count :
            ]
            - self.taxa_count
        )

    def _call(self, x):
        return transform_ratios(x, self.tree.bounds, self.tree.preorder)

    def _inverse(self, y):
        return transform_heights_to_ratios(y, self.tree.bounds, self.tree.preorder)

    def log_abs_det_jacobian(self, x, y):
        return torch.log(
            y[..., self.indices_sorted] - self.tree.bounds[self.taxa_count : -1]
        ).sum(-1)


def heights_to_branch_lengths(node_heights, bounds, indexing):
    taxa_count = int((bounds.shape[0] + 1) / 2)
    indices_sorted = indexing[np.argsort(indexing[:, 1])].transpose()
    return torch.cat(
        (
            node_heights[..., indices_sorted[0, :taxa_count] - taxa_count]
            - bounds[:taxa_count],
            node_heights[..., indices_sorted[0, taxa_count:] - taxa_count]
            - node_heights[..., indices_sorted[1, taxa_count:] - taxa_count],
        ),
        -1,
    )


def transform_ratios(ratios_root_height: torch.Tensor, bounds, indexing):
    """Transform node ratios/root height to internal node heights.

    :param root_height: root height
    :param ratios: node ratios of internal nodes
    :param bounds: lower bound of each node
    :param indexing: pairs of parent/child indices, must be preorder or inorder
    :return: internal node heights
    """
    taxa_count = ratios_root_height.shape[-1] + 1
    heights = ratios_root_height.clone()
    for parent_id, id_ in indexing:
        if id_ >= taxa_count:
            heights[..., id_ - taxa_count] = bounds[id_] + ratios_root_height[
                ..., id_ - taxa_count
            ] * (heights[..., parent_id - taxa_count] - bounds[id_])
    return heights


def transform_heights_to_ratios(
    node_heights: torch.Tensor, bounds: torch.Tensor, indexing
):
    """Transform internal node heights to ratios/root height.

    :param node_heights: internal node heights
    :param bounds: lower bound of each node
    :param indexing: pairs of parent/child indices, must be preorder or inorder
    :return: node ratios of internal nodes
    """
    taxa_count = int((bounds.shape[0] + 1) / 2)
    indices_sorted = indexing[np.argsort(indexing[:, 1])].transpose()
    return torch.cat(
        (
            (
                node_heights[..., indices_sorted[1, taxa_count:] - taxa_count]
                - bounds[taxa_count:-1]
            )
            / (
                node_heights[..., indices_sorted[0, taxa_count:] - taxa_count]
                + bounds[taxa_count:-1]
            ),
            node_heights[..., -1:],
        )
    )


def setup_indexes(tree):
    for node in tree.postorder_node_iter():
        node.index = -1
        node.annotations.add_bound_attribute("index")

    indexer = iter(range(len(tree.taxon_namespace), len(tree.taxon_namespace) * 2 - 1))
    taxa_dict = {taxon.label: idx for idx, taxon in enumerate(tree.taxon_namespace)}

    for node in tree.postorder_node_iter():
        if not node.is_leaf():
            node.index = next(indexer)
        else:
            node.index = taxa_dict[node.taxon.label]


def setup_dates(tree, heterochronous=False):
    # parse dates
    if heterochronous:
        dates = {}
        for node in tree.leaf_node_iter():
            dates[str(node.taxon)] = float(str(node.taxon).rsplit('_', 1)[:-1])

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


def initialize_dates_from_taxa(tree, taxa, tag='date'):
    dates = [taxon[tag] for taxon in taxa]
    max_date = max(dates)

    # parse dates
    if max_date != 0.0:
        # time starts at 0
        if min(dates) == 0.0:
            for node in tree.leaf_node_iter():
                node.date = taxa[node.index][tag]
                node.original_date = node.date
        # time is a year
        else:
            for node in tree.leaf_node_iter():
                node.date = max_date - taxa[node.index][tag]
                node.original_date = taxa[node.index][tag]
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
    return heights[len(tree.taxon_namespace) :]


def parse_tree(taxa, data):
    taxon_namespace = TaxonNamespace([taxon.id for taxon in taxa])
    taxon_namespace_size = len(taxon_namespace)
    if 'newick' in data:
        tree = Tree.get(
            data=data['newick'],
            schema='newick',
            preserve_underscores=True,
            rooting='force-rooted',
            taxon_namespace=taxon_namespace,
        )
    elif 'file' in data:
        tree = Tree.get(
            path=data['file'],
            schema='newick',
            preserve_underscores=True,
            rooting='force-rooted',
            taxon_namespace=taxon_namespace,
        )
    else:
        raise ValueError('Tree model requires a file or newick element to be specified')
    if taxon_namespace_size != len(taxon_namespace):
        raise ValueError(
            'Some taxon names in the tree do not match those in the Taxa object'
        )
    tree.resolve_polytomies(update_bipartitions=True)
    setup_indexes(tree)
    return tree


class TreeModel(Model):
    _tag = 'tree_model'

    @abc.abstractmethod
    def branch_lengths(self) -> torch.Tensor:
        ...

    @property
    @abc.abstractmethod
    def postorder(self) -> List[List[int]]:
        ...

    @property
    @abc.abstractmethod
    def taxa(self) -> List[str]:
        ...

    @abc.abstractmethod
    def write_newick(self, steam, **kwargs) -> None:
        ...


class AbstractTreeModel(TreeModel, ABC):
    def __init__(self, id_: ID, tree, taxa: Taxa) -> None:
        TreeModel.__init__(self, id_)
        self.tree = tree
        self._taxa = taxa
        self.taxa_count = len(tree.taxon_namespace)
        self._postorder = []
        self.update_traversals()

    def update_traversals(self) -> None:
        # postorder for peeling
        self._postorder = []
        for node in self.tree.postorder_node_iter():
            if not node.is_leaf():
                children = node.child_nodes()
                self._postorder.append(
                    (node.index, children[0].index, children[1].index)
                )

    def handle_model_changed(self, model, obj, index):
        pass

    @property
    def postorder(self):
        return self._postorder

    @property
    def taxa(self):
        # return [taxon.label for taxon in self.tree.taxon_namespace]
        return [taxon.id for taxon in self._taxa]

    def as_newick(self, **kwargs):
        out = StringIO()
        self.write_newick(out, **kwargs)
        return out.getvalue()

    def write_newick(self, steam, **kwargs) -> None:
        self._write_newick(self.tree.seed_node, steam, **kwargs)

    def _write_newick(self, node, steam, **kwargs) -> None:
        if not node.is_leaf():
            steam.write('(')
            for i, child in enumerate(node.child_node_iter()):
                self._write_newick(child, steam, **kwargs)
                if i == 0:
                    steam.write(',')
            steam.write(')')
        else:
            taxon_index = kwargs.get('taxon_index', None)
            if not taxon_index:
                steam.write(str(node.taxon).strip("'"))
            else:
                steam.write(str(node.index + 1))
        if node.parent_node is not None:
            branch_lengths = kwargs.get('branch_lengths', self.branch_lengths())
            steam.write(':{}'.format(branch_lengths[node.index]))
        else:
            steam.write(';')


class UnRootedTreeModel(AbstractTreeModel):
    def __init__(self, id_: ID, tree, taxa: Taxa, branch_lengths: Parameter) -> None:
        super().__init__(id_, tree, taxa)
        self._branch_lengths = branch_lengths
        self.add_parameter(branch_lengths)

    def branch_lengths(self) -> torch.Tensor:
        return self._branch_lengths.tensor

    @property
    def sample_shape(self) -> torch.Size:
        return self._branch_lengths.tensor.shape[:-1]

    def update(self, value):
        if isinstance(value, dict):
            if self._branch_lengths.id in value:
                self._branch_lengths.tensor = value[self._branch_lengths.id]
        else:
            self._branch_lengths.tensor = value

    def handle_parameter_changed(self, variable, index, event):
        self.fire_model_changed()

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        taxa = process_object(data['taxa'], dic)
        tree = parse_tree(taxa, data)
        branch_lengths = process_object(data['branch_lengths'], dic)
        if 'keep_branch_lengths' in data:
            branch_lengths.tensor = torch.tensor(
                torch.tensor(
                    [
                        float(node.edge_length)
                        for node in sorted(
                            list(tree.postorder_node_iter())[:-1], key=lambda x: x.index
                        )
                    ],
                    dtype=branch_lengths.dtype,
                )
            )
        return cls(id_, tree, taxa, branch_lengths)


class TimeTreeModel(AbstractTreeModel):
    def __init__(self, id_: ID, tree, taxa: Taxa, node_heights: Parameter) -> None:
        super().__init__(id_, tree, taxa)
        self._node_heights = node_heights
        self.taxa_count = len(tree.taxon_namespace)
        self.bounds = None
        self.sampling_times = None
        self.update_leaf_heights()
        self.update_bounds()
        if node_heights is not None:
            self.add_parameter(node_heights)
        self._branch_lengths = None
        self.branch_lengths_need_update = True

    def update_leaf_heights(self, dtype=torch.float64) -> None:
        leaf_heights = [None] * len(self._taxa)

        dates = [taxon['date'] for taxon in self._taxa]
        max_date = max(dates)

        # time starts at 0
        if min(dates) == 0.0:
            for idx, taxon in enumerate(self._taxa):
                leaf_heights[idx] = taxon['date']
        # time is a year
        else:
            for idx, taxon in enumerate(self._taxa):
                leaf_heights[idx] = max_date - taxon['date']

        self.sampling_times = torch.tensor(leaf_heights, dtype=dtype)

    def update_bounds(self) -> None:
        taxa_count = self.taxa_count
        internal_heights = [None] * (taxa_count - 1)
        for node, left, right in self.postorder:
            left_height = (
                self.sampling_times[left]
                if left < taxa_count
                else internal_heights[left - taxa_count]
            )
            right_height = (
                self.sampling_times[right]
                if right < taxa_count
                else internal_heights[right - taxa_count]
            )

            internal_heights[node - taxa_count] = (
                left_height if left_height > right_height else right_height
            )
        self.bounds = torch.cat(
            (self.sampling_times, torch.stack(internal_heights)), -1
        )

    def update_traversals(self):
        super().update_traversals()
        # preoder indexing to go from ratios to heights
        self.preorder = np.array(
            [
                (node.parent_node.index, node.index)
                for node in self.tree.preorder_node_iter()
                if node != self.tree.seed_node
            ]
        )

    @property
    def node_heights(self) -> torch.Tensor:
        return self._node_heights.tensor

    def branch_lengths(self) -> torch.Tensor:
        """Return branch lengths calculated from node heights.

        Branch lengths are indexed by node index on the distal side of
        the tree. For example branch_lengths[0] corresponds to the branch
        starting from taxon with index 0.

        :return: branch lengths of tree
        :rtype: torch.Tensor
        """
        if self.branch_lengths_need_update:
            indices_sorted = self.preorder[np.argsort(self.preorder[:, 1])].transpose()
            heights = torch.cat(
                (
                    self.sampling_times.expand(
                        self._node_heights.tensor.shape[:-1] + (-1,)
                    ),
                    self._node_heights.tensor,
                ),
                -1,
            )
            self._branch_lengths = (
                heights[..., indices_sorted[0]] - heights[..., indices_sorted[1]]
            )
            self.branch_lengths_need_update = False
        return self._branch_lengths

    def update(self, value):
        if isinstance(value, dict):
            if self._branch_lengths.id in value:
                self._branch_lengths.tensor = value[self._branch_lengths.id]
        else:
            self._branch_lengths.tensor = value

    def handle_parameter_changed(self, variable, index, event):
        self.branch_lengths_need_update = True
        self.fire_model_changed()

    @property
    def sample_shape(self) -> torch.Size:
        return self._node_heights.tensor.shape[:-1]

    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> None:
        super().cuda(device)
        self.bounds.cuda(device)

    def cpu(self) -> None:
        super().cpu()
        self.bounds.cpu()

    @staticmethod
    def json_factory(id_: str, newick: str, taxa: Union[dict, list, str], **kwargs):
        r"""
        Factory for creating tree models in JSON format.

        :param id_: ID of the tree model
        :param newick: tree in newick format
        :param taxa: dictionary of taxa with attributes or str reference


        :key node_heights_id:  ID of node_heights
        :key node_heights: node_heights. Can be a list of floats, a dictionary
        corresponding to a transformed parameter, or a str corresponding to a reference

        :return: tree model in JSON format compatible with from_json class method
        """

        tree_model = {
            'id': id_,
            'type': 'phylotorch.evolution.tree_model.TimeTreeModel',
            'newick': newick,
        }
        node_heights = kwargs.get('node_heights', None)
        node_heights_id = kwargs.get('node_heights_id', None)
        if node_heights is None:
            tree_model['node_heights'] = {"id": node_heights_id}
        elif isinstance(node_heights, list):
            tree_model['node_heights'] = {
                "id": node_heights_id,
                "type": "phylotorch.Parameter",
                "tensor": node_heights,
            }
        elif isinstance(node_heights, (dict, str)):
            tree_model['node_heights'] = node_heights

        if isinstance(taxa, dict):
            taxon_list = []
            for taxon in taxa.keys():
                taxon_list.append(
                    {
                        "id": taxon,
                        "type": "phylotorch.evolution.taxa.Taxon",
                        "attributes": {"date": taxa[taxon]},
                    }
                )
            tree_model['taxa'] = {
                'id': 'taxa',
                'type': 'phylotorch.evolution.taxa.Taxa',
                'taxa': taxon_list,
            }
        elif isinstance(taxa, list):
            tree_model['taxa'] = {
                'id': 'taxa',
                'type': 'phylotorch.evolution.taxa.Taxa',
                'taxa': taxa,
            }
        else:
            tree_model['taxa'] = taxa

        return tree_model

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        taxa = process_object(data['taxa'], dic)
        tree = parse_tree(taxa, data)
        initialize_dates_from_taxa(tree, taxa)

        if isinstance(data['node_heights'], dict) and len(data['node_heights']) == 1:
            # TODO: get rid of this. Define a Parameter and use a flag like
            #       `keep_heights=true` to populate the Parameter from the tree.
            #       Special care if node_height parameter is a transformed parameter
            node_heights_id = data['node_heights']['id']
            heights_np = heights_from_branch_lengths(tree)
            node_heights = Parameter(node_heights_id, torch.tensor(heights_np))
            dic[node_heights_id] = node_heights
            tree_model = cls(id_, tree, taxa, node_heights)
        else:
            # TODO: tree_model and node_heights may have circular references to each
            #       other when node_heights is a transformed Parameter requiring
            #       the tree_model
            tree_model = cls(id_, tree, taxa, None)
            dic[id_] = tree_model
            tree_model._node_heights = process_object(data['node_heights'], dic)
            tree_model.add_parameter(tree_model._node_heights)

        return tree_model
