import os

import pytest

from phylotorch.evolution.substmodel import JC69

data_dir = 'data'


@pytest.fixture
def tiny_newick_file():
    return os.path.join(data_dir, 'tiny.nwk')


@pytest.fixture
def tiny_fasta_file():
    return os.path.join(data_dir, 'tiny.fa')


@pytest.fixture
def flu_a_tree_file():
    return os.path.join(data_dir, 'fluA.tree')


@pytest.fixture
def flu_a_fasta_file():
    return os.path.join(data_dir, 'fluA.fa')


@pytest.fixture
def jc69_model():
    return JC69('jc')
