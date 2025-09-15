from __future__ import annotations

import re
import sys
from collections import OrderedDict

import numpy as np
import torch
import treezy.fasta as fasta
from treezy import NewickReader, NexusReader, Tree

from .tree import setup_dates, setup_indexes


def parse_tree(taxon_names, *, newick=None, file=None, **kwargs):
    if newick is not None:
        tree = Tree.from_newick(
            newick,
            taxon_names
            # preserve_underscores=True,
            # rooting='force-rooted',
        )
    elif file is not None:
        tree_format = 'newick'
        with open(file) as fp:
            if next(fp).upper().startswith('#NEXUS'):
                tree_format = 'nexus'
        if tree_format == 'nexus':
            with NexusReader(file, taxon_names) as reader:
                tree = reader.next()
        else:
            with NewickReader(file, taxon_names) as reader:
                tree = reader.next()
    else:
        raise ValueError('Tree model requires a file or newick element to be specified')

    tree.make_binary()
    use_postorder_indices = kwargs.get('use_postorder_indices', False)
    setup_indexes(tree, use_postorder_indices)
    return tree


def read_tree(tree, dated=True, heterochornous=True):
    tree = parse_tree(None, file=tree)

    if dated:
        setup_dates(tree, heterochornous)
    return tree


def read_tree_and_alignment(tree, alignment, dated=True, heterochornous=True):
    tree = read_tree(tree, dated, heterochornous)
    dna = fasta.parse_to_dict(alignment)
    dna = OrderedDict((k, dna[k]) for k in tree.taxon_names)
    sequence_count = len(dna)
    if sequence_count != len(dna):
        sys.stderr.write('taxon names in trees and alignment are different')
        exit(2)
    return tree, dna


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


def random_tree_from_heights(sampling: torch.Tensor, heights: torch.Tensor) -> Node:
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
