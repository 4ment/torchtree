from typing import List, Union

try:
    from typing import OrderedDict  # introduced in the python 3.7.2

    OrderedDictType = OrderedDict
except ImportError:
    from typing import MutableMapping

    OrderedDictType = MutableMapping
import torch

from . import Parameter

ListTensor = List[torch.Tensor]
ListParameter = List[Parameter]
ID = Union[str, None]
