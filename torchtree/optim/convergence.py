from __future__ import annotations

import abc
import collections
import math
import sys

import numpy as np
import torch

from ..core.model import CallableModel
from ..core.serializable import JSONSerializable
from ..core.utils import JSONParseError, process_objects, register_class


class BaseConvergence(JSONSerializable):
    """Base class for all convergence diagnostic classes."""

    @abc.abstractmethod
    def check(self, iteration: int, *args, **kwargs) -> bool:
        pass


@register_class
class VariationalConvergence(BaseConvergence):
    """Class that does not check for convergence but output ELBO.

    :param loss: ELBO function
    :param int every: evaluate ELBO at every "every" iteration
    :param torch.Size samples: number of samples for ELBO calculation
    :param int start: start checking at iteration number "start (Default is 0)"
    :param str file_name: print to file_name or print to sys.stdout if file_name is None
    """

    def __init__(
        self,
        loss: CallableModel,
        every: int,
        samples: torch.Size,
        start: int = 0,
        file_name: str = None,
    ) -> None:
        self.loss = loss
        self.every = every
        self.samples = samples
        self.start = start
        self.file_name = file_name
        if self.file_name is not None:
            self.f = open(self.file_name, 'w')
        else:
            self.f = sys.stdout

    def check(self, iteration: int, *args, **kwargs) -> bool:
        if iteration >= self.start and iteration % self.every == 0:
            if self.samples == torch.Size([0]):
                elbo = self.loss.lp
            else:
                with torch.no_grad():
                    elbo = self.loss(samples=self.samples)
            self.f.write('{} ELBO {}\n'.format(iteration, elbo))
            self.f.flush()
        return True

    @classmethod
    def from_json(
        cls, data: dict[str, any], dic: dict[str, any]
    ) -> VariationalConvergence:
        loss = process_objects(data['loss'], dic)
        every = data.get('every', 100)
        samples = data.get('samples', 100)
        if isinstance(samples, list):
            samples = torch.Size(samples)
        else:
            samples = torch.Size((samples,))
        start = data.get('start', 0)
        optionals = {}
        if 'file_name' in data:
            optionals['file_name'] = data['file_name']
        return cls(loss, every, samples, start, **optionals)


@register_class
class StanVariationalConvergence(VariationalConvergence):
    """Class for checking SGD convergence using Stan's algorithm.

    Code adapted from:
     https://github.com/stan-dev/stan/blob/develop/src/stan/variational/advi.hpp

    :param CallableModel loss: ELBO function
    :param int every: evaluate ELBO at every "every" iteration
    :param int samples: number of samples for ELBO calculation
    :param int max_iterations: maximum number of iterations
    :param int start: start checking at iteration number "start" (Default is 0)
    :param float tol_rel_obj: relative tolerance parameter for convergence
     (Default is 0.01)
    """

    def __init__(
        self,
        loss: CallableModel,
        every: int,
        samples: torch.Size,
        max_iterations: int,
        start: int = 0,
        tol_rel_obj: float = 0.01,
    ) -> None:
        super().__init__(loss, every, samples, start)
        self.tol_rel_obj = tol_rel_obj
        self.elbo = 0.0
        self.elbo_best = -sys.float_info.max
        self.elbo_prev = -sys.float_info.max
        self.delta_elbo = -sys.float_info.max
        self.delta_elbo_ave = -sys.float_info.max
        self.delta_elbo_med = -sys.float_info.max
        # Heuristic to estimate how far to look back in rolling window
        cb_size = int(max(0.1 * max_iterations / every, 2.0))
        self.elbo_diff = collections.deque(maxlen=cb_size)
        print('  iter             ELBO   delta_ELBO_mean   delta_ELBO_med   notes ')

    def check(self, iteration: int, *args, **kwargs) -> bool:
        keep_going = True
        if iteration >= self.start and iteration % self.every == 0:
            elbo_prev = self.elbo
            if self.samples == torch.Size([0]):
                self.elbo = self.loss.lp.item()
            else:
                with torch.no_grad():
                    self.elbo = self.loss(samples=self.samples).item()
            if self.elbo > self.elbo_best:
                self.elbo_best = self.elbo
            delta_elbo = StanVariationalConvergence.rel_difference(self.elbo, elbo_prev)
            self.elbo_diff.append(delta_elbo)
            delta_elbo_ave = np.cumsum(self.elbo_diff)[-1] / len(self.elbo_diff)
            delta_elbo_med = np.median(list(self.elbo_diff))
            ss = ''
            if delta_elbo_ave < self.tol_rel_obj:
                ss = '   MEAN ELBO CONVERGED'
                keep_going = False

            if delta_elbo_med < self.tol_rel_obj:
                ss = '   MEDIAN ELBO CONVERGED'
                keep_going = False

            if iteration > 10 * self.every:
                if delta_elbo_med > 0.5 or delta_elbo_ave > 0.5:
                    ss += '   MAY BE DIVERGING... INSPECT ELBO'

            print(
                '  {:>4}  {:>15.3f}  {:>16.3f}  {:>15.3f}{}'.format(
                    iteration, self.elbo, delta_elbo_ave, delta_elbo_med, ss
                )
            )
        return keep_going

    @staticmethod
    def rel_difference(prev: float, curr: float) -> float:
        """Compute the relative difference between two double values.

        :param prev: previous value
        :param curr: current value
        :return: absolutely value of relative difference
        """
        return math.fabs((curr - prev) / prev)

    @classmethod
    def from_json(
        cls, data: dict[str, any], dic: dict[str, any]
    ) -> StanVariationalConvergence:
        loss = process_objects(data['loss'], dic)
        every = data.get('every', 100)
        samples = data.get('samples', 100)
        if isinstance(samples, list):
            samples = torch.Size(samples)
        else:
            samples = torch.Size((samples,))
        start = data.get('start', 0)
        if 'max_iterations' in data:
            max_iterations = data['max_iterations']
        else:
            raise JSONParseError(
                'StanVBConvergence needs max_iterations to be specified'
            )
        tol_rel_obj = data.get('tol_rel_obj', 0.01)
        return cls(loss, every, samples, max_iterations, start, tol_rel_obj)
