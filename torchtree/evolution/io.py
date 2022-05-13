from __future__ import annotations

import re
import sys

import dendropy
import numpy as np
import torch
from dendropy import DnaCharacterMatrix, Tree

from .tree_model import setup_dates, setup_indexes


def read_tree(tree, dated=True, heterochornous=True):
    taxa = dendropy.TaxonNamespace()
    tree_format = 'newick'
    with open(tree) as fp:
        if next(fp).upper().startswith('#NEXUS'):
            tree_format = 'nexus'

    tree = Tree.get(
        path=tree,
        schema=tree_format,
        tree_offset=0,
        taxon_namespace=taxa,
        preserve_underscores=True,
        rooting='force-rooted',
    )
    tree.resolve_polytomies(update_bipartitions=True)

    setup_indexes(tree)
    if dated:
        setup_dates(tree, heterochornous)
    return tree


def read_tree_and_alignment(tree, alignment, dated=True, heterochornous=True):
    tree = read_tree(tree, dated, heterochornous)

    # alignment
    seqs_args = dict(schema='nexus', preserve_underscores=True)
    with open(alignment) as fp:
        if next(fp).startswith('>'):
            seqs_args = dict(schema='fasta')
    dna = DnaCharacterMatrix.get(
        path=alignment, taxon_namespace=tree.taxon_namespace, **seqs_args
    )
    sequence_count = len(dna)
    if sequence_count != len(dna.taxon_namespace):
        sys.stderr.write('taxon names in trees and alignment are different')
        exit(2)
    return tree, dna


def to_nexus(node, fp):
    if not node.is_leaf():
        fp.write('(')
        for i, n in enumerate(node.child_node_iter()):
            to_nexus(n, fp)
            if i == 0:
                fp.write(',')
        fp.write(')')
    else:
        fp.write(str(node.index + 1))
    if hasattr(node, 'date'):
        fp.write('[&height={}'.format(node.date))
        if hasattr(node, 'rate'):
            fp.write(',rate={}'.format(node.rate))
        fp.write(']')
    if node.parent_node is not None:
        fp.write(':{}'.format(node.edge_length))
    else:
        fp.write(';')


def convert_samples_to_nexus(tree, sample, output):
    taxaCount = len(tree.taxon_namespace)
    outp = open(output, 'w')
    outp.write('#NEXUS\nBegin trees;\nTranslate\n')
    outp.write(
        ',\n'.join(
            [
                str(i + 1) + ' ' + x.label.replace("'", '')
                for i, x in enumerate(tree.taxon_namespace)
            ]
        )
    )
    outp.write('\n;\n')

    for i in range(len(sample)):
        for n in tree.postorder_node_iter():
            if not n.is_leaf():
                n.date = float(sample[i][n.index - taxaCount])
        for n in tree.postorder_node_iter():
            if n.parent_node is not None:
                n.edge_length = n.parent_node.date - n.date
        outp.write('tree {} = '.format(i + 1))
        to_nexus(tree.seed_node, outp)
        outp.write('\n')
    outp.write('END;')
    outp.close()


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


def parse_translate(fp):
    taxa = []
    for line in fp:
        if line.strip() == ';':
            break
        line2 = line.replace(',', '').replace(';', '').strip()
        _, taxon = re.search(r"(\d+)\s+([^,]+)[,;]?", line2).groups()
        taxa.append(taxon)
        if ';' in line:
            break
    return taxa


def parse_trees(fp, count=None):
    trees = []
    taxa = []
    for line in fp:
        line = line.strip()
        if line.lower().startswith('tree'):
            chunks = re.split(r"\s+", line)
            if len(taxa) == 0:
                newick_split = split_newick(chunks[-1])
                taxa = filter(lambda x: x[0] not in '[](),;:', newick_split)
                taxa = list(map(lambda x: x.split(':')[0], taxa))
            if count == 0:
                break
        if line.lower().startswith('translate'):
            taxa = parse_translate(fp)
            if count == 0:
                break

    return taxa, trees


def parse_taxa(fp):
    ntax_regex = re.compile(r"\s*dimensions\s+ntax\s*=\s*(\d+)", re.IGNORECASE)
    taxlabel_regex = re.compile(r"\s*taxlabels(.*)", re.IGNORECASE)
    all = ''
    for line in fp:
        line = line.strip()
        ntax_match = ntax_regex.match(line)
        taxlabel_regex_match = taxlabel_regex.match(line)
        if ntax_match:
            ntax = ntax_match.group(1)
        if taxlabel_regex_match:
            all += taxlabel_regex_match.group(1)
            break

    while ';' not in all:
        all += next(fp).strip()
    taxa = re.split(r"\s+", all.replace(';', '').strip())
    assert taxa == ntax
    return taxa


def split_newick(newick: str) -> list[str]:
    """Split tree in newick format around (),;

    Example:

        >>> newick = '((a:1[&a={1,2}],b:1):1,c:1);'
        >>> split_newick('((a:1,b:1):1,c:1);')
        ['(', '(', 'a:1', ',', 'b:1', ')', ':1', ',', 'c:1', ')', ';']

    :param str newick: newick tree
    :return List[str]: list of strings
    """
    chunks = re.split(r"(\[[^\]]+\])", newick)
    newick_split = []
    for c in chunks:
        if c.startswith('['):
            newick_split.append(c)
        elif c != '':
            chunks2 = re.split(r"([\(\),;])", c)
            newick_split.extend(chunks2)
    return list(filter(None, newick_split))


def extract_taxa(file_name: str) -> list[str]:
    """Extract taxon list from a nexus file.

    This function will try get the taxon names from the taxa and trees blocks.

    :param str file_name: path to the nexus file
    :return List[str]: list of taxon names
    """
    taxa = None
    tree_format = 'newick'
    REGEX_TAXA = re.compile(r"begin\s+taxa;", re.IGNORECASE)
    REGEX_TREES = re.compile(r"begin\s+trees;", re.IGNORECASE)
    with open(file_name) as fp:
        if next(fp).upper().startswith('#NEXUS'):
            tree_format = 'nexus'

    if tree_format == 'newick':
        with open(file_name) as fp:
            newick = next(fp).strip()
        newick_split = split_newick(newick)
        taxa = filter(lambda x: x[0] not in '[](),;:', newick_split)
        taxa = list(map(lambda x: x.split(':')[0], taxa))
    else:
        with open(file_name) as fp:
            for line in fp:
                line = line.strip()
                m = REGEX_TAXA.match(line)
                trees_match = REGEX_TREES.match(line)
                if m:
                    taxa = parse_taxa(fp)
                    break

                if trees_match:
                    taxa, _ = parse_trees(fp, 0)
                    break

    return taxa
