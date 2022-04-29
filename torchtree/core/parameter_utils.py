from __future__ import annotations

import json
import os

from torchtree.core.parameter import Parameter
from torchtree.core.parameter_encoder import ParameterEncoder


def save_parameters(file_name: str, parameters: list[Parameter], safely=True):
    r"""Save a list of parameters to a json file.

    :param str file_name: output file path
    :param parameters: list of parameters
    :type parameters: list(Parameter)
    :param bool safely: Create a temporary file if True
    """
    if not safely or not os.path.lexists(file_name):
        # for var_name in self.optimizer.state_dict():
        #     print(var_name, "\t", self.optimizer.state_dict()[var_name])
        # torch.save(self.optimizer.state_dict(), 'checkpoint.json')
        with open(file_name, 'w') as fp:
            json.dump(parameters, fp, cls=ParameterEncoder, indent=2)
    else:
        # torch.save(self.optimizer.state_dict(), 'checkpoint-new.json')
        with open(file_name + '.new', 'w') as fp:
            json.dump(parameters, fp, cls=ParameterEncoder, indent=2)
        os.rename(file_name, file_name + '.old')
        os.rename(file_name + '.new', file_name)
        os.remove(file_name + '.old')
