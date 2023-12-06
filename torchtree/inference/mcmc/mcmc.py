from __future__ import annotations

from typing import Any, List

import torch

from torchtree.core.identifiable import Identifiable
from torchtree.core.model import CallableModel
from torchtree.core.parameter_utils import save_parameters
from torchtree.core.runnable import Runnable
from torchtree.core.utils import (
    SignalHandler,
    process_object,
    process_objects,
    register_class,
)
from torchtree.inference.mcmc.operator import MCMCOperator
from torchtree.typing import ID


@register_class
class MCMC(Identifiable, Runnable):
    def __init__(
        self,
        id_: ID,
        joint: CallableModel,
        operators: List[MCMCOperator],
        iterations: int,
        **kwargs,
    ) -> None:
        Identifiable.__init__(self, id_)
        self._operators = operators
        self.joint = joint
        self.iterations = iterations
        self.loggers = kwargs.get("loggers", ())
        self.checkpoint = kwargs.get("checkpoint", None)
        self.checkpoint_frequency = kwargs.get("checkpoint_frequency", 1000)
        self.every = kwargs.get("every", 100)
        self._epoch = 1
        self.parameters = []
        for op in operators:
            for parameter in op.parameters:
                self.parameters.extend(parameter.parameters())

    def run(self) -> None:
        accept = 0

        for logger in self.loggers:
            logger.initialize()
            logger.log(sample=0)

        with torch.no_grad():
            log_joint = self.joint()

        handler = SignalHandler()

        if self.every != 0:
            print("  iter             logP   accept ratio   step size ")
            print(f"  {0:>4}  {log_joint:>15.3f}")

        while self._epoch <= self.iterations:
            if handler.stop:
                break

            weights = torch.tensor([operator.weight for operator in self._operators])
            index_operator = torch.distributions.Categorical(weights).sample().item()
            operator = self._operators[index_operator]

            hastings_ratio = operator.step()

            if torch.isinf(hastings_ratio):
                log_alpha = torch.finfo(hastings_ratio.dtype).min
                acceptance_prob = torch.zeros_like(hastings_ratio)
                accepted = False
            else:
                with torch.no_grad():
                    log_joint_proposed = self.joint()
                if torch.isnan(log_joint_proposed):
                    log_alpha = torch.finfo(hastings_ratio.dtype).min
                    acceptance_prob = torch.zeros_like(hastings_ratio)
                    accepted = False
                else:
                    log_alpha = (log_joint_proposed - log_joint) + hastings_ratio
                    acceptance_prob = min(torch.zeros(1), log_alpha).exp()
                    accepted = (acceptance_prob > torch.rand(1)).item()

            if accepted:
                log_joint = log_joint_proposed
                accept += 1
                operator.accept()
            else:
                operator.reject()

            if self.every != 0 and self._epoch % self.every == 0:
                step_size = 0
                if hasattr(operator, "_integrator"):
                    step_size = operator._integrator.step_size

                print(
                    f"  {self._epoch:>4}  {log_joint:>15.3f}"
                    f" {accept / self._epoch:>13.3f} {step_size:>11.3e}"
                )

            for logger in self.loggers:
                logger.log(sample=self._epoch)

            operator.tune(acceptance_prob, sample=self._epoch, accepted=accepted)

            if (
                self.checkpoint is not None
                and self._epoch % self.checkpoint_frequency == 0
            ):
                self.save_full_state()

            self._epoch += 1

            if self.checkpoint is not None and epoch % self.checkpoint_frequency == 0:
                save_parameters(self.checkpoint, self.parameters)

        for logger in self.loggers:
            logger.close()

        for op in self._operators:
            print(
                op.id,
                op._accept / (op._accept + op._reject),
                op._accept + op._reject,
                op.tuning_parameter,
            )

    def state_dict(self) -> dict[str, Any]:
        states = {"iteration": self._epoch}
        states["operators"] = [op.state_dict() for op in self._operators]
        return states

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for op in self._operators:
            for op_state in state_dict["operators"]:
                if op.id == op_state["id"]:
                    op.load_state_dict(op_state)
                    break
        self._epoch = state_dict["iteration"]

    def save_full_state(self) -> None:
        mcmc_state = {
            "id": self.id,
            "type": "MCMC",
        }
        mcmc_state.update(self.state_dict())
        full_state = [mcmc_state] + self.parameters
        save_parameters(self.checkpoint, full_state)

    @classmethod
    def from_json(cls, data: dict[str, Any], dic: dict[str, Any]) -> MCMC:
        iterations = data["iterations"]

        optionals = {}
        # checkpointing is used by default and the default file name is checkpoint.json
        # it can be disabled if "checkpoint": false is used
        # the name of the checkpoint file can be modified using
        # "checkpoint": "checkpointer.json"
        if "checkpoint" in data:
            if isinstance(data["checkpoint"], bool) and data["checkpoint"]:
                optionals["checkpoint"] = "checkpoint.json"
            elif isinstance(data["checkpoint"], str):
                optionals["checkpoint"] = data["checkpoint"]
        else:
            optionals["checkpoint"] = "checkpoint.json"

        if "checkpoint_frequency" in data:
            optionals["checkpoint_frequency"] = data["checkpoint_frequency"]

        if "loggers" in data:
            loggers = process_objects(data["loggers"], dic)
            if not isinstance(loggers, list):
                loggers = list(loggers)
            optionals["loggers"] = loggers

        joint = process_object(data["joint"], dic)

        operators = process_objects(data["operators"], dic)

        if "every" in data:
            optionals["every"] = data["every"]

        return cls(data["id"], joint, operators, iterations, **optionals)
