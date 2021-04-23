import copy
import inspect
import json
import os
from typing import Dict, Union, Tuple

from torch.optim import Optimizer as TorchOptimizer

from ..core.model import Parameter, Parametric, CallableModel
from ..core.parameter_encoder import ParameterEncoder
from ..core.runnable import Runnable
from ..core.serializable import JSONSerializable
from ..core.utils import get_class, process_objects, SignalHandler, JSONParseError
from ..typing import ListParameter, ListTensor


class Optimizer(JSONSerializable, Runnable):
    """
    A wrapper for torch.optim.Optimizer objects.

    :param parameters: list of Parameter
    :param loss: loss function
    :param optimizer: a torch.optim.Optimizer object
    :param iterations: number of iterations
    :param kwargs:
    """

    def __init__(self, parameters: ListParameter, loss: CallableModel, optimizer: TorchOptimizer, iterations: int,
                 **kwargs):
        self.parameters = parameters
        self.loss = loss
        self.optimizer = optimizer
        self.iterations = iterations
        self.scheduler = kwargs.get('scheduler', None)
        self.convergence = kwargs.get('convergence', None)
        self.loggers = kwargs.get('loggers', ())
        self.maximize = kwargs.get('maximize', True)

    def run(self) -> None:
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

            if epoch % 1000 == 0:
                if not os.path.lexists('checkpoint.json'):
                    # for var_name in self.optimizer.state_dict():
                    #     print(var_name, "\t", self.optimizer.state_dict()[var_name])
                    # torch.save(self.optimizer.state_dict(), 'checkpoint.json')
                    with open('checkpoint.json', 'w') as fp:
                        json.dump(self.parameters, fp, cls=ParameterEncoder, indent=2)
                else:
                    # torch.save(self.optimizer.state_dict(), 'checkpoint-new.json')
                    with open('checkpoint-new.json', 'w') as fp:
                        json.dump(self.parameters, fp, cls=ParameterEncoder, indent=2)
                    os.rename('checkpoint.json', 'checkpoint-old.json')
                    os.rename('checkpoint-new.json', 'checkpoint.json')
                    os.remove('checkpoint-old.json')

        for logger in self.loggers:
            if hasattr(logger, 'finalize'):
                logger.finalize()

    @staticmethod
    def parse_params(params: Dict[str, Union[list, float]], dic) -> Tuple[ListTensor, ListParameter]:
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
                    'Optimizable expects a list of Parameters or Parametric models\n{} was provided'.format(type(poml)))
        return tensors, parameters

    @classmethod
    def from_json(cls, data: Dict[str, any], dic: Dict[str, any]) -> 'Optimizer':
        rules = {
            'loss': {'type': 'object|string', 'instanceof': 'CallableModel'},
            'parameters': {'type': 'object|string', 'instanceof': 'Parameter', 'list': True},
            'iterations': {'type': 'int', 'constraint': {'>': 0}},
            'algorithm': {'type': 'string', 'instanceof': 'torch.optim.Optimizer'},
            'maximize': {'type': 'bool', 'optional': True},
            'loggers': {'type': 'object', 'instanceof': 'Logger', 'list': True, 'optional': True},
            'scheduler': {'type': 'object', 'instanceof': 'Scheduler', 'optional': True},
            'lr': {'type': 'numbers.Number', 'constraint': {'>': 0.0}, 'optional': True},
            'convergence': {'type': 'object', 'optional': True}
        }
        # validate(data, rules)

        iterations = data['iterations']

        optionals = {}

        if 'loggers' in data:
            loggers = process_objects(data["loggers"], dic)
            if not isinstance(loggers, list):
                loggers = list(loggers)
            optionals['loggers'] = loggers

        loss = process_objects(data['loss'], dic)

        optimizer_params = []  # an iterable of torch.Tensor s or dict s. Specifies what Tensors should be optimized
        parameters = []  # a list of phylotorch.Parameters
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
        optimizer = optim_class(optimizer_params, **kwargs)  # type: int

        # instanciate torch.optim.scheduler
        if 'scheduler' in data:
            klass = get_class(data['scheduler']['type'])
            optionals['scheduler'] = klass.from_json(data['scheduler'], None, optimizer=optimizer)

        # instanciate phylotorch.optim.Convergence
        if 'convergence' in data:
            klass = get_class(data['convergence']['type'])
            convergence = klass.from_json(data['convergence'], dic)
            optionals['convergence'] = convergence

        return cls(parameters, loss, optimizer, iterations, **optionals)
