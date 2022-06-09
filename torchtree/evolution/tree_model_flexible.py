from typing import Union

from ..core.utils import JSONParseError, process_object, register_class
from .tree_model import (
    TimeTreeModel,
    heights_from_branch_lengths,
    initialize_dates_from_taxa,
    parse_tree,
)


@register_class
class FlexibleTimeTreeModel(TimeTreeModel):
    @staticmethod
    def json_factory(
        id_: str,
        newick: str,
        internal_heights: Union[dict, list, str],
        taxa: Union[dict, list, str],
        **kwargs
    ):
        r"""
        Factory for creating tree models in JSON format.

        :param id_: ID of the tree model
        :param newick: tree in newick format
        :param taxa: dictionary of taxa with attributes or str reference


        :key internal_heights_id:  ID of internal_heights
        :key internal_heights: internal node heights. Can be a list of floats,
        a dictionary corresponding to a transformed parameter, or a str corresponding
        to a reference

        :return: tree model in JSON format compatible with from_json class method
        """

        tree_model = {
            'id': id_,
            'type': 'FlexibleTimeTreeModel',
            'newick': newick,
        }
        if 'keep_branch_lengths' in kwargs and kwargs['keep_branch_lengths']:
            tree_model['keep_branch_lengths'] = kwargs['keep_branch_lengths']

        node_heights_id = kwargs.get('internal_heights_id', None)
        if isinstance(internal_heights, list):
            tree_model['internal_heights'] = {
                "id": node_heights_id,
                "type": "torchtree.Parameter",
                "tensor": internal_heights,
            }
        elif isinstance(internal_heights, (dict, str)):
            tree_model['internal_heights'] = internal_heights

        if isinstance(taxa, dict):
            taxon_list = []
            for taxon in taxa.keys():
                taxon_list.append(
                    {
                        "id": taxon,
                        "type": "torchtree.evolution.taxa.Taxon",
                        "attributes": {"date": taxa[taxon]},
                    }
                )
            tree_model['taxa'] = {
                'id': kwargs.get('taxa_id', 'taxa'),
                'type': 'torchtree.evolution.taxa.Taxa',
                'taxa': taxon_list,
            }
        elif isinstance(taxa, list):
            tree_model['taxa'] = {
                'id': kwargs.get('taxa_id', 'taxa'),
                'type': 'torchtree.evolution.taxa.Taxa',
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

        # TODO: tree_model and internal_heights may have circular references to each
        #       other when internal_heights is a transformed Parameter requiring
        #       the tree_model
        if id_ in dic:
            raise JSONParseError('Object with ID `{}\' already exists'.format(id_))
        tree_model = cls(id_, tree, taxa, None)
        dic[id_] = tree_model
        tree_model._internal_heights = process_object(data['internal_heights'], dic)

        if data.get('keep_branch_lengths', False):
            tree_model._internal_heights.tensor = heights_from_branch_lengths(tree).to(
                dtype=tree_model._internal_heights.dtype
            )

        return tree_model
