import sys

import dendropy
from dendropy import Tree, DnaCharacterMatrix

from .evolution.tree_model import setup_indexes, setup_dates


def read_tree_and_alignment(tree, alignment, dated=True, heterochornous=True):
    # tree
    taxa = dendropy.TaxonNamespace()
    tree_format = 'newick'
    with open(tree) as fp:
        if next(fp).upper().startswith('#NEXUS'):
            tree_format = 'nexus'

    tree = Tree.get(path=tree, schema=tree_format, tree_offset=0, taxon_namespace=taxa, preserve_underscores=True,
                    rooting='force-rooted')
    tree.resolve_polytomies(update_bipartitions=True)

    setup_indexes(tree)
    if dated:
        setup_dates(tree, heterochornous)

    # alignment
    seqs_args = dict(schema='nexus', preserve_underscores=True)
    with open(alignment) as fp:
        if next(fp).startswith('>'):
            seqs_args = dict(schema='fasta')
    dna = DnaCharacterMatrix.get(path=alignment, taxon_namespace=taxa, **seqs_args)
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
    outp.write(',\n'.join([str(i + 1) + ' ' + x.label.replace("'", '') for i, x in enumerate(tree.taxon_namespace)]))
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
