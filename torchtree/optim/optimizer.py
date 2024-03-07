from __future__ import annotations

import copy
import inspect
from typing import Any

import torch
from torch.optim import Optimizer as TorchOptimizer

from torchtree.core.identifiable import Identifiable
from torchtree.core.model import CallableModel
from torchtree.core.parameter_utils import save_parameters
from torchtree.core.runnable import Runnable
from torchtree.core.utils import (
    JSONParseError,
    SignalHandler,
    get_class,
    process_objects,
    register_class,
)
from torchtree.inference.utils import extract_tensors_and_parameters
from torchtree.typing import ID, ListParameter


@register_class
class Optimizer(Identifiable, Runnable):
    r"""A wrapper for torch.optim.Optimizer objects.

    :param list parameters: list of Parameter
    :param CallableModel loss: loss function
    :param optimizer: a torch.optim.Optimizer object
    :type optimizer: torch.optim.Optimizer
    :param int iterations: number of iterations
    :param kwargs: optionals
    """

    def __init__(
        self,
        id_: ID,
        parameters: ListParameter,
        loss: CallableModel,
        optimizer: TorchOptimizer,
        iterations: int,
        **kwargs,
    ) -> None:
        Identifiable.__init__(self, id_)
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
        self.checkpoint_all = kwargs.get('checkpoint_all', False)
        self.distributions = kwargs.get('distributions', None)
        self._epoch = 1

    def _run_closure(self) -> None:
        def closure():
            for p in self.parameters:
                p.fire_parameter_changed()
            loss = -self.loss() if self.maximize else self.loss()
            self.optimizer.zero_grad()
            loss.backward()
            return loss

        handler = SignalHandler()

        loss = self.loss()
        print(f"{0:>4} {loss:.5f}")

        for p in self.parameters:
            p.requires_grad = True

        while self._epoch <= self.iterations:
            if handler.stop:
                break
            self.optimizer.step(closure)
            state = self.optimizer.state_dict()['state'][0]

            with torch.no_grad():
                loss = self.loss()

            func_evals = state["func_evals"]
            n_iter = state["n_iter"]
            print(f"{n_iter:>4} {loss:.5f} evaluations: {func_evals}")

            if (
                self.checkpoint is not None
                and self._epoch % self.checkpoint_frequency == 0
            ):
                if self.checkpoint_all:
                    checkpoint_file = self.checkpoint.replace(
                        ".json", f"-{self._epoch}.json"
                    )
                    self.save_full_state(checkpoint_file, overwrite=True)
                else:
                    self.save_full_state(self.checkpoint)

            self._epoch += 1

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

        while self._epoch <= self.iterations:
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
                logger(self._epoch)

            if self.convergence is not None:
                if self.distributions:
                    for distr in self.distributions:
                        distr.sample(self.convergence.samples)
                res = self.convergence.check(self._epoch)
                if not res:
                    break
                # x of the variational distribution has changed due to sampling
                for p in self.parameters:
                    p.fire_parameter_changed()

            if (
                self.checkpoint is not None
                and self._epoch % self.checkpoint_frequency == 0
            ):
                if self.checkpoint_all:
                    checkpoint_file = self.checkpoint.replace(
                        ".json", f"-{self._epoch}.json"
                    )
                    self.save_full_state(checkpoint_file, overwrite=True)
                else:
                    self.save_full_state(self.checkpoint)

            self._epoch += 1

        for logger in self.loggers:
            logger.close()

    def run(self) -> None:
        if isinstance(self.optimizer, torch.optim.LBFGS):
            self._run_closure()
        else:
            self._run()

    def state_dict(self) -> dict[str, Any]:
        return {"iteration": self._epoch, "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._epoch = state_dict["iteration"]
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def save_full_state(self, checkpoint, safely=True, overwrite=False) -> None:
        optimizer_state = {
            "id": self.id,
            "type": "Optimizer",
        }
        optimizer_state.update(self.state_dict())
        full_state = [optimizer_state] + self.parameters
        save_parameters(checkpoint, full_state, safely, overwrite)

    @classmethod
    def from_json(cls, data: dict[str, Any], dic: dict[str, Any]) -> Optimizer:
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
        if "checkpoint_all" in data:
            optionals["checkpoint_all"] = data["checkpoint_all"]

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
        optim_options = data.get("options", {})
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

        return cls(data["id"], parameters, loss, optimizer, iterations, **optionals)
