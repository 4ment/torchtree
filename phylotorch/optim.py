import abc
import inspect

import torch

from .core.model import Parameter
from .core.runnable import Runnable
from .core.serializable import JSONSerializable
from .core.utils import get_class, process_objects, SignalHandler, validate


class Convergence(JSONSerializable):

    @abc.abstractmethod
    def check(self, iterations):
        pass


class StanConvergence(Convergence):

    def __init__(self, loss, every, samples):
        self.loss = loss
        self.every = every
        self.samples = samples

    def check(self, iterations):
        if iterations % self.every == 0:
            with torch.no_grad():
                elbo = self.loss(samples=self.samples)
            # TODO: implement
            print(iterations, 'ELBO', elbo)
        return True

    @classmethod
    def from_json(cls, data, dic):
        loss = process_objects(data['loss'], dic)
        every = data.get('every', 100)
        samples = data.get('samples', 100)
        return cls(loss, every, samples)


class Optimizer(JSONSerializable, Runnable):

    def __init__(self, parameters, loss, optimizer, iterations, **kwargs):
        self.parameters = parameters
        self.loss = loss
        self.optimizer = optimizer
        self.iterations = iterations
        self.scheduler = kwargs.get('scheduler', None)
        self.convergence = kwargs.get('convergence', None)
        self.loggers = kwargs.get('loggers', ())
        self.maximize = kwargs.get('maximize', True)

    def run(self):
        for logger in self.loggers:
            if hasattr(logger, 'init'):
                logger.init()

        handler = SignalHandler()
        if self.convergence is not None:
            self.convergence.check(0)

        for p in self.parameters:
            p.requires_grad = True

        for epoch in range(1, self.iterations):
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

        for logger in self.loggers:
            if hasattr(logger, 'finalize'):
                logger.finalize()

    @classmethod
    def from_json(cls, data, dic):
        rules = {
            'loss': {'type': 'object|string', 'instanceof': 'CallableModel'},
            'parameters': {'type': 'object|string', 'instanceof': 'Parameter', 'list': True},
            'iterations': {'type': 'int', 'constraint': {'>': 0}},
            'algorithm': {'type': 'string', 'instanceof': 'torch.optim.Optimizer'},
            'maximize': {'type': 'bool', 'optional': True},
            'loggers': {'type': 'object', 'instanceof': 'Scheduler', 'list': True, 'optional': True},
            'scheduler': {'type': 'object', 'instanceof': 'Scheduler', 'optional': True},
            'lr': {'type': 'numbers.Number', 'constraint': {'>': 0.0}, 'optional': True},
            'convergence': {'type': 'object', 'optional': True}
        }
        validate(data, rules)

        iterations = data['iterations']

        optionals = {}

        if 'loggers' in data:
            loggers = process_objects(data["loggers"], dic)
            if not isinstance(loggers, list):
                loggers = list(loggers)
            optionals['loggers'] = loggers

        # if 'loggers' in data:
        #     loggers = [process_object(logger, dic) for logger in data["loggers"]]
        # loggers = list(map(Logger.from_json, data["loggers"]))

        loss = process_objects(data['loss'], dic)
        parameters = process_objects(data['parameters'], dic)

        # instanciate torch.optim.optimizer
        optim_class = get_class(data['algorithm'])
        tensors = [p.tensor for p in parameters]
        signature_params = list(inspect.signature(optim_class.__init__).parameters)
        kwargs = {}
        for arg in signature_params[1:]:
            if arg in data:
                kwargs[arg] = data[arg]
        optimizer = optim_class(tensors, **kwargs)

        # instanciate torch.optim.scheduler
        if 'scheduler' in data:
            klass = get_class(data['scheduler']['type'])
            kwargs = {}
            if 'lr_lambda' in data['scheduler']:
                kwargs['lr_lambda'] = eval(data['scheduler']['lr_lambda'])
            scheduler = klass(optimizer, **kwargs)
            optionals['scheduler'] = scheduler

        # instanciate phylotorch.optim.Convergence
        if 'convergence' in data:
            klass = get_class(data['convergence']['type'])
            convergence = klass.from_json(data['convergence'], dic)
            optionals['convergence'] = convergence

        return cls(parameters, loss, optimizer, iterations, **optionals)


class Scheduler(JSONSerializable):

    @classmethod
    def from_json(cls, data, dic):
        klass = get_class(data['algorithm'])

        signature_params = inspect.signature(klass.__init__).parameters
        params = []
        params.append(data['optimizer'])
        # lr_lambda = lambda epoch: 1.0 / np.sqrt(epoch + 1)
        if 'lr_lambda' in data:
            params.append(eval(data['lr_lambda']))
        # for arg in signature_params[2:]:
        #     if arg in data:
        #         params.append(eval(data[arg]))

        return klass(*params)
