:py:mod:`torchtree.optim.convergence`
=====================================

.. py:module:: torchtree.optim.convergence


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   torchtree.optim.convergence.BaseConvergence
   torchtree.optim.convergence.VariationalConvergence
   torchtree.optim.convergence.StanVariationalConvergence




.. py:class:: BaseConvergence



   Base class for all convergence diagnostic classes.

   .. py:method:: check(iteration: int, *args, **kwargs) -> bool
      :abstractmethod:



.. py:class:: VariationalConvergence(loss: torchtree.core.model.CallableModel, every: int, samples: torch.Size, start: int = 0, file_name: str = None)



   Class that does not check for convergence but output ELBO.

   :param loss: ELBO function
   :param int every: evaluate ELBO at every "every" iteration
   :param torch.Size samples: number of samples for ELBO calculation
   :param int start: start checking at iteration number "start (Default is 0)"
   :param str file_name: print to file_name or print to sys.stdout if file_name is None

   .. py:method:: check(iteration: int, *args, **kwargs) -> bool


   .. py:method:: from_json(data: dict[str, any], dic: dict[str, any]) -> VariationalConvergence
      :classmethod:



.. py:class:: StanVariationalConvergence(loss: torchtree.core.model.CallableModel, every: int, samples: torch.Size, max_iterations: int, start: int = 0, tol_rel_obj: float = 0.01)



   Class for checking SGD convergence using Stan's algorithm.

   Code adapted from:
    https://github.com/stan-dev/stan/blob/develop/src/stan/variational/advi.hpp

   :param CallableModel loss: ELBO function
   :param int every: evaluate ELBO at every "every" iteration
   :param int samples: number of samples for ELBO calculation
   :param int max_iterations: maximum number of iterations
   :param int start: start checking at iteration number "start" (Default is 0)
   :param float tol_rel_obj: relative tolerance parameter for convergence
    (Default is 0.01)

   .. py:method:: check(iteration: int, *args, **kwargs) -> bool


   .. py:method:: rel_difference(prev: float, curr: float) -> float
      :staticmethod:

      Compute the relative difference between two double values.

      :param prev: previous value
      :param curr: current value
      :return: absolutely value of relative difference


   .. py:method:: from_json(data: dict[str, any], dic: dict[str, any]) -> StanVariationalConvergence
      :classmethod:



