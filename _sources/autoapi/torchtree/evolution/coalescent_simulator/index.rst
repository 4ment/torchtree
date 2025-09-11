torchtree.evolution.coalescent_simulator
========================================

.. py:module:: torchtree.evolution.coalescent_simulator


Functions
---------

.. autoapisummary::

   torchtree.evolution.coalescent_simulator.sample_coalescent_times
   torchtree.evolution.coalescent_simulator.sample_tree


Module Contents
---------------

.. py:function:: sample_coalescent_times(sampling_times: torch.Tensor, sampling_counts: torch.Tensor, trajectory, lower_bound: float)

   Simulate from inhomogeneous, heterochronous coalescent using thinning algorithm.

   # Code adapted from https://github.com/mdkarcher/phylodyn/R/coalsieve.R

   :param Tensor sampling_times: one-dimentional tensor containing sampling times.
   :param Tensor sampling_counts: samples taken per sampling time.
   :param trajectory: function that returns effective population size at time t.
   :param float lower_bound: lower limit of trajectory function on its support.
   :return: coalescent times.
   :rtype: Tensor

   :example:
   >>> _ = torch.manual_seed(0)
   >>> unif_traj = lambda t: 1.0
   >>> sample_coalescent_times(torch.arange(3.0), torch.tensor([3,2,1]), unif_traj, lower_bound=0.1)
   tensor([0.1169, 0.6442, 1.6912, 1.7371, 2.0807])


.. py:function:: sample_tree(sampling_times: torch.Tensor, sampling_counts: torch.Tensor, coalescent_times: torch.Tensor)

   Generate a tree from coalescent times, sampling times, and sampling counts.

   :param Tensor sampling_times: sampling times.
   :param Tensor sampling_counts: samples taken per sampling time.
   :param Tensor coalescent_times: coalescent times.
   :return: A dendropy tree object.


