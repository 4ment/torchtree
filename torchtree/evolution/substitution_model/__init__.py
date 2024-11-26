r"""Markov substitution process.

The Markov substitution process is a stochastic model used to describe the evolution of sequences (such as nucleotides in DNA or amino acids in proteins) along the branches of a phylogenetic tree.
This process is based on the principle of Markov chains, where the state of a system at a given time depends only on its immediate previous state.
In the context of phylogenetics, the states typically correspond to the nucleotide or character states at each site in the sequence.

In this model, the substitution process is described by a **rate matrix** (also called the **Q matrix**), which specifies the instantaneous rates at which one state can change to another.

1. **Markov Process Basics**
----------------------------

The substitution process is modeled as a continuous-time Markov process. The probability of observing a sequence of substitutions between nucleotides (or other characters) over time is governed by a set of transition rates, which are typically represented in a rate matrix :math:`Q`.

The rate matrix :math:`Q` is defined as a square matrix where the off-diagonal elements represent the rate of substitution between different states (nucleotides or characters), and the diagonal elements are negative, ensuring the rows sum to zero.

For a simple example of nucleotide substitutions with 4 states \{A,C,G,T\}, the rate matrix :math:`Q` might look like:

.. math::

    Q = \begin{bmatrix}
    -3\mu & \mu & \mu & \mu \\
    \mu & -3\mu & \mu & \mu \\
    \mu & \mu & -3\mu & \mu \\
    \mu & \mu & \mu & -3\mu
    \end{bmatrix}

where :math:`\mu` represents the rate of substitution between different nucleotide pairs.

2. **Probability Matrix and Time Evolution**
--------------------------------------------

The probability of transitioning between states over a given time period :math:`t` is given by the **probability matrix** :math:`P(t)`, which is related to the rate matrix :math:`Q` by the matrix exponential:

.. math::

    P(t) = e^{Qt}

This matrix describes the probability of moving from one nucleotide (or character) to another after time :math:`t`, based on the rates defined in :math:`Q`.

The probability matrix is essential for calculating the likelihood of a given sequence alignment under a particular evolutionary model.
The transition probabilities can be computed over each branch of a phylogenetic tree, and the tree can be estimated by maximizing the likelihood of the observed data under the chosen model.

3. **Stationary Distribution**
------------------------------

In some cases, the Markov process reaches a stationary distribution, where the probabilities of being in each state become constant over time.
For nucleotide substitution models, the stationary distribution often represents the equilibrium base frequencies of the nucleotides.

For example, if the stationary distribution for the nucleotide 'A' is :math:`\pi_A`, then the probability of observing 'A' at equilibrium will be :math:`\pi_A`, and similarly for the other nucleotides.

4. **Markov Substitution in Phylogenetics**
-------------------------------------------

In phylogenetics, the Markov substitution process is used to model how nucleotide or character states evolve along the branches of a phylogenetic tree.
The phylogenetic tree represents the relationships between species, and branch lengths represent the amount of evolutionary time or the number of substitutions.

Given a sequence alignment and a model of nucleotide substitution (such as the Jukes-Cantor or GTR models), the goal is to estimate the evolutionary history of the species (the tree) and the rates of substitution along the branches.
The Markov substitution model allows the computation of the likelihood of observing the given sequences under the model, and the tree that maximizes this likelihood is considered the most probable evolutionary tree.

"""
from torchtree.evolution.substitution_model.amino_acid import LG, WAG
from torchtree.evolution.substitution_model.codon import MG94
from torchtree.evolution.substitution_model.general import (
    EmpiricalSubstitutionModel,
    GeneralJC69,
    GeneralNonSymmetricSubstitutionModel,
    GeneralSymmetricSubstitutionModel,
)
from torchtree.evolution.substitution_model.nucleotide import GTR, HKY, JC69

__all__ = [
    'JC69',
    'HKY',
    'GTR',
    'EmpiricalSubstitutionModel',
    'GeneralJC69',
    'GeneralSymmetricSubstitutionModel',
    'GeneralNonSymmetricSubstitutionModel',
    'LG',
    'WAG',
    'MG94',
]
