from __future__ import annotations

import copy
import inspect

import torch
from torch.optim import Optimizer as TorchOptimizer

from torchtree.inference.utils import extract_tensors_and_parameters

from ..core.model import CallableModel
from ..core.parameter_utils import save_parameters
from ..core.runnable import Runnable
from ..core.serializable import JSONSerializable
from ..core.utils import (
    JSONParseError,
    SignalHandler,
    get_class,
    process_objects,
    register_class,
)
from ..typing import ListParameter


@register_class
class Optimizer(JSONSerializable, Runnable):
    """A wrapper for torch.optim.Optimizer objects.

    :param list parameters: list of Parameter
    :param CallableModel loss: loss function
    :param torch.optim.Optimizer optimizer: a torch.optim.Optimizer object
    :param int iterations: number of iterations
    :param kwargs: optionals
    """

    def __init__(
        self,
        parameters: ListParameter,
        loss: CallableModel,
        optimizer: TorchOptimizer,
        iterations: int,
        **kwargs,
    ) -> None:
        self.parameters = parameters
        self.loss = loss
        self.optimizer = optimizer
        self.iterations = iterations
        self.scheduler = kwargs.get('scheduler', None)
        self.convergence = kwargs.get('convergence', None)
        self.loggers = kwargs.get('loggers', ())
        self.maximize = kwargs.get('maximize', True)
        self.checkpoint = kwargs.get('checkpoint', None)
        self.checkpoint_frequency = kwargs.get('checkpoint_frequency', 1000)
        self.distributions = kwargs.get('distributions', None)

    def _run_closure(self) -> None:
        def closure():
            for p in self.parameters:
                p.fire_parameter_changed()
            loss = -self.loss() if self.maximize else self.loss()
            self.optimizer.zero_grad()
            loss.backward()
            return loss

        handler = SignalHandler()

        for p in self.parameters:
            p.requires_grad = True

        for epoch in range(self.iterations):
            if handler.stop:
                break
            self.optimizer.step(closure)
            state = self.optimizer.state_dict()['state'][0]

            with torch.no_grad():
                loss = self.loss()

            func_evals = state["func_evals"]
            n_iter = state["n_iter"]
            print(f'{n_iter:>4} {loss} evaluations: {func_evals}')

            save_parameters(self.checkpoint, self.parameters)

    def _run(self) -> None:
        for logger in self.loggers:
            if hasattr(logger, 'init'):
                logger.init()

        handler = SignalHandler()
        if self.convergence is not None:
            if self.distributions:
                for distr in self.distributions:
                    distr.sample(self.convergence.samples)
            self.convergence.check(0)
            for p in self.parameters:
                p.fire_parameter_changed()

        for p in self.parameters:
            p.requires_grad = True

        trials = 10

        for epoch in range(1, self.iterations + 1):
            if handler.stop:
                break

            for trial in range(trials):
                if self.distributions:
                    for distr in self.distributions:
                        distr.sample(self.loss.samples)

                loss = -self.loss() if self.maximize else self.loss()

                self.optimizer.zero_grad()
                loss.backward()

                for p in self.parameters:
                    retry = torch.any(torch.isinf(p.grad)) or torch.any(
                        torch.isnan(p.grad)
                    )
                    if retry:
                        break
                if not retry:
                    break
                else:
                    for p in self.parameters:
                        p.fire_parameter_changed()

            self.optimizer.step()

            for p in self.parameters:
                p.fire_parameter_changed()

            if self.scheduler is not None:
                self.scheduler.step()

            for logger in self.loggers:
                logger(epoch)

            if self.convergence is not None:
                if self.distributions:
                    for distr in self.distributions:
                        distr.sample(self.convergence.samples)
                res = self.convergence.check(epoch)
                if not res:
                    break
                # x of the variational distribution has changed due to sampling
                for p in self.parameters:
                    p.fire_parameter_changed()

            if self.checkpoint is not None and epoch % self.checkpoint_frequency == 0:
                save_parameters(self.checkpoint, self.parameters)

        for logger in self.loggers:
            if hasattr(logger, 'finalize'):
                logger.finalize()

    def run(self) -> None:
        if isinstance(self.optimizer, torch.optim.LBFGS):
            self._run_closure()
        else:
            self._run()

    @classmethod
    def from_json(cls, data: dict[str, any], dic: dict[str, any]) -> Optimizer:
        rules = {  # noqa: F841
            'loss': {'type': 'object|string', 'instanceof': 'CallableModel'},
            'parameters': {
                'type': 'object|string',
                'instanceof': 'Parameter',
                'list': True,
            },
            'iterations': {'type': 'int', 'constraint': {'>': 0}},
            'algorithm': {'type': 'string', 'instanceof': 'torch.optim.Optimizer'},
            'maximize': {'type': 'bool', 'optional': True},
            'loggers': {
                'type': 'object',
                'instanceof': 'Logger',
                'list': True,
                'optional': True,
            },
            'scheduler': {
                'type': 'object',
                'instanceof': 'Scheduler',
                'optional': True,
            },
            'lr': {
                'type': 'numbers.Number',
                'constraint': {'>': 0.0},
                'optional': True,
            },
            'convergence': {'type': 'object', 'optional': True},
        }
        # validate(data, rules)

        iterations = data['iterations']

        optionals = {}
        # checkpointing is used by default and the default file name is checkpoint.json
        # it can be disabled if 'checkpoint': false is used
        # the name of the checkpoint file can be modified using
        # 'checkpoint': 'checkpointer.json'
        if 'checkpoint' in data:
            if isinstance(data['checkpoint'], bool) and data['checkpoint']:
                optionals['checkpoint'] = 'checkpoint.json'
            elif isinstance(data['checkpoint'], str):
                optionals['checkpoint'] = data['checkpoint']
        else:
            optionals['checkpoint'] = 'checkpoint.json'

        if 'checkpoint_frequency' in data:
            optionals['checkpoint_frequency'] = data['checkpoint_frequency']

        if 'loggers' in data:
            loggers = process_objects(data["loggers"], dic)
            if not isinstance(loggers, list):
                loggers = list(loggers)
            optionals['loggers'] = loggers

        loss = process_objects(data['loss'], dic)

        # an iterable of torch.Tensors or dicts: specifies what Tensors should
        # be optimized
        optimizer_params = []
        parameters = []  # a list of torchtree.Parameters
        if isinstance(data['parameters'][0], dict):
            # [{'params': ['model']},
            #  {'params': ["parameter1", "parameter2"], 'lr': 1e-3}
            # ], lr=1e-2, momentum=0.9
            for param_groups in data['parameters']:
                t, p = extract_tensors_and_parameters(param_groups['params'], dic)
                param_groups_cpy = copy.deepcopy(param_groups)
                param_groups_cpy['params'] = t
                optimizer_params.append(param_groups_cpy)
                parameters.extend(p)

        else:
            # "parameters": ["model", "parameter1", "parameter2"]
            optimizer_params, parameters = extract_tensors_and_parameters(
                data['parameters'], dic
            )

        # instanciate torch.optim.optimizer
        optim_class = get_class(data['algorithm'])
        optim_options = data['options']
        signature_params = list(inspect.signature(optim_class.__init__).parameters)
        # 'maximize' is specified in some Optimizers (e.g. Adam but not LBFGS)
        # from pytorch 1.11. Here we never provide maximize to Adam and
        # use the maximize option in torchtree.Optimizer instead.
        if 'maximize' in optim_options:
            optionals['maximize'] = optim_options['maximize']
            del optim_options['maximize']
        elif 'maximize' in data:
            optionals['maximize'] = data['maximize']
        else:
            optionals['maximize'] = True

        for option in optim_options:
            if option not in signature_params[1:]:
                raise JSONParseError(
                    f'{option} is not a valid option for {optim_class.__name__}'
                    ' optimizer\n'
                    'Choose from: ' + ','.join(signature_params[1:]) + '\n'
                    'https://pytorch.org/docs/stable/optim.html'
                )
        optimizer = optim_class(optimizer_params, **optim_options)

        # instanciate torch.optim.scheduler
        if 'scheduler' in data:
            klass = get_class(data['scheduler']['type'])
            optionals['scheduler'] = klass.from_json(
                data['scheduler'], None, optimizer=optimizer
            )

        # instanciate torchtree.optim.Convergence
        if 'convergence' in data:
            klass = get_class(data['convergence']['type'])
            convergence = klass.from_json(data['convergence'], dic)
            optionals['convergence'] = convergence

        # these distributions will be sampled before the loss is calculated
        # these distributions are considered to be fixed (no gradient calculation)
        if 'distributions' in data:
            optionals['distributions'] = process_objects(data["distributions"], dic)

        return cls(parameters, loss, optimizer, iterations, **optionals)
