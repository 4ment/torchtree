import copy
import inspect
import json
import os
from typing import Dict, Tuple, Union

import torch
from torch.optim import Optimizer as TorchOptimizer

from .. import Parameter
from ..core.model import CallableModel, Parametric
from ..core.parameter_encoder import ParameterEncoder
from ..core.runnable import Runnable
from ..core.serializable import JSONSerializable
from ..core.utils import (
    JSONParseError,
    SignalHandler,
    get_class,
    process_objects,
    register_class,
)
from ..typing import ListParameter, ListTensor


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

    def update_checkpoint(self):
        if not os.path.lexists(self.checkpoint):
            # for var_name in self.optimizer.state_dict():
            #     print(var_name, "\t", self.optimizer.state_dict()[var_name])
            # torch.save(self.optimizer.state_dict(), 'checkpoint.json')
            with open(self.checkpoint, 'w') as fp:
                json.dump(self.parameters, fp, cls=ParameterEncoder, indent=2)
        else:
            # torch.save(self.optimizer.state_dict(), 'checkpoint-new.json')
            with open(self.checkpoint + '.new', 'w') as fp:
                json.dump(self.parameters, fp, cls=ParameterEncoder, indent=2)
            os.rename(self.checkpoint, self.checkpoint + '.old')
            os.rename(self.checkpoint + '.new', self.checkpoint)
            os.remove(self.checkpoint + '.old')

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

            self.update_checkpoint()

    def _run(self) -> None:
        for logger in self.loggers:
            if hasattr(logger, 'init'):
                logger.init()

        handler = SignalHandler()
        if self.convergence is not None:
            self.convergence.check(0)

        for p in self.parameters:
            p.requires_grad = True

        for epoch in range(1, self.iterations + 1):
            if handler.stop:
                break

            loss = -self.loss() if self.maximize else self.loss()

            # print(-loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            for p in self.parameters:
                p.fire_parameter_changed()

            if self.scheduler is not None:
                self.scheduler.step()

            for logger in self.loggers:
                logger(epoch)

            if self.convergence is not None:
                res = self.convergence.check(epoch)
                if not res:
                    break

            if self.checkpoint is not None and epoch % 1000 == 0:
                self.update_checkpoint()

        for logger in self.loggers:
            if hasattr(logger, 'finalize'):
                logger.finalize()

    def run(self) -> None:
        if isinstance(self.optimizer, torch.optim.LBFGS):
            self._run_closure()
        else:
            self._run()

    @staticmethod
    def parse_params(
        params: Dict[str, Union[list, float]], dic
    ) -> Tuple[ListTensor, ListParameter]:
        parameter_or_parametric_list = process_objects(params, dic)
        if not isinstance(parameter_or_parametric_list, list):
            parameter_or_parametric_list = [parameter_or_parametric_list]

        tensors = []
        parameters = []
        for poml in parameter_or_parametric_list:
            if isinstance(poml, Parameter):
                tensors.append(poml.tensor)
                parameters.append(poml)
            elif isinstance(poml, Parametric):
                for parameter in poml.parameters():
                    tensors.append(parameter.tensor)
                    parameters.append(parameter)
            else:
                raise JSONParseError(
                    'Optimizable expects a list of Parameters or Parametric models\n{}'
                    ' was provided'.format(type(poml))
                )
        return tensors, parameters

    @classmethod
    def from_json(cls, data: Dict[str, any], dic: Dict[str, any]) -> 'Optimizer':
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

        if 'maximize' in data:
            optionals['maximize'] = data['maximize']

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
                t, p = cls.parse_params(param_groups['params'], dic)
                param_groups_cpy = copy.deepcopy(param_groups)
                param_groups_cpy['params'] = t
                optimizer_params.append(param_groups_cpy)
                parameters.extend(p)

        else:
            # "parameters": ["model", "parameter1", "parameter2"]
            optimizer_params, parameters = cls.parse_params(data['parameters'], dic)

        # instanciate torch.optim.optimizer
        optim_class = get_class(data['algorithm'])
        signature_params = list(inspect.signature(optim_class.__init__).parameters)
        kwargs = {}
        for arg in signature_params[1:]:
            if arg in data:
                kwargs[arg] = data[arg]
        optimizer = optim_class(optimizer_params, **kwargs)

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

        return cls(parameters, loss, optimizer, iterations, **optionals)
