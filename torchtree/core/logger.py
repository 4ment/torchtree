from __future__ import annotations

import csv
import json
import sys
from abc import abstractmethod
from typing import Union

import torch

from ..evolution.tree_model import TreeModel
from .abstractparameter import AbstractParameter
from .model import CallableModel
from .parameter_encoder import ParameterEncoder
from .runnable import Runnable
from .serializable import JSONSerializable
from .utils import process_object, process_objects, register_class


class LoggerInterface(JSONSerializable, Runnable):
    """Interface for logging things like parameters or trees to a file."""

    @abstractmethod
    def log(self, *args, **kwargs) -> None:
        ...

    @abstractmethod
    def initialize(self) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    def run(self) -> None:
        self.initialize()
        self.log()
        self.close()


@register_class
class Logger(LoggerInterface):
    r"""
    Class for logging Parameter objects to a file.

    :param objs: list of Parameter or CallableModel objects
    :type objs: list[Parameter or CallableModel]
    :param int every: logging frequency
    :param kwargs: optionals
    """

    def __init__(
        self, objs: list[Union[AbstractParameter, CallableModel]], every: int, **kwargs
    ) -> None:
        if 'file_name' in kwargs:
            self.file_name = kwargs['file_name']
            del kwargs['file_name']
        else:
            self.file_name = None
        self.every = every
        self.kwargs = kwargs
        self.objs = objs
        self.f = None
        self.writer = None
        self.sample = 1

    def initialize(self) -> None:
        if self.file_name:
            self.f = open(self.file_name, 'w')
        else:
            self.f = sys.stdout
        self.writer = csv.writer(self.f, **self.kwargs)
        header = ['sample']
        for obj in self.objs:
            if isinstance(obj, AbstractParameter):
                header.extend(['{}.{}'.format(obj.id, i) for i in range(obj.shape[-1])])

            else:
                header.append(obj.id)
        self.writer.writerow(header)

    def log(self, *args, **kwargs) -> None:
        sample = kwargs.get('sample', self.sample)
        self.sample += 1
        if sample % self.every != 0:
            return

        row = [sample]
        for obj in self.objs:
            if isinstance(obj, AbstractParameter):
                row.extend(obj.tensor.detach().cpu().tolist())
            else:
                log_p = obj()
                if len(log_p.shape) == 0 or log_p.shape[-1] <= 1:
                    row.append(log_p.item())
                else:
                    row.append(log_p.sum(-1).item())
        self.writer.writerow(row)

    def close(self) -> None:
        if self.file_name is not None:
            self.f.close()

    @classmethod
    def from_json(cls, data, dic) -> Logger:
        r"""Create a Logger object.

        :param data: json representation of Logger object.
        :type data: dict[str,Any]
        :param dic: dictionary containing additional objects that can be referenced
         in data.
        :type dic: dict[str,Any]

        :return: a :class:`~torchtree.core.logger.Logger` object.
        :rtype: Logger
        """
        params = process_objects(data['parameters'], dic)
        every = data.get('every', 1)
        kwargs = {}
        for key in ('file_name', 'delimiter'):
            if key in data:
                kwargs[key] = data[key]
        return cls(params, every, **kwargs)


@register_class
class TreeLogger(LoggerInterface):
    """Class for logging trees to a file.

    :param TreeModel objs: TreeModel object
    :param int every: logging frequency
    :param kwargs: optionals
    """

    def __init__(self, tree_model: TreeModel, every: int, **kwargs) -> None:
        self.tree_model = tree_model
        self.file_name = kwargs.get('file_name', None)
        self.kwargs = kwargs
        self.sample = 1
        self.f = None

    def initialize(self) -> None:
        if self.file_name is not None:
            self.f = open(self.file_name, 'w')
        else:
            self.f = sys.stdout
        if self.kwargs.get('format', 'newick') == 'nexus':
            self.f.write('#NEXUS\nBegin trees;\nTranslate\n')
            self.f.write(
                ',\n'.join(
                    [
                        str(i + 1) + ' ' + x.replace("'", '')
                        for i, x in enumerate(self.tree_model.taxa)
                    ]
                )
            )
            self.f.write('\n;\n')

    def log(self, *args, **kwargs) -> None:
        sample = kwargs.get('sample', self.sample)
        self.sample += 1
        if sample % self.every != 0:
            return

        tree_format = self.kwargs.get('format', 'newick')
        if tree_format == 'newick':
            self.tree_model.write_newick(self.f)
        else:
            self.f.write('tree {} = '.format(sample))
            optionals = {'taxon_index': True}  # replace taxon name by its index
            self.tree_model.write_newick(self.f, **optionals)
        self.f.write('\n')

    def close(self) -> None:
        if self.kwargs.get('format', 'newick') == 'nexus':
            self.f.write('\nEND;')
        if self.file_name is not None:
            self.f.close()

    @classmethod
    def from_json(cls, data, dic) -> TreeLogger:
        r"""
        Create a TreeLogger object.

        :param data: json representation of TreeLogger object.
        :type data: dict[str,Any]
        :param dic: dictionary containing additional objects that can be referenced
         in data.
        :type dic: dict[str,Any]

        :return: a :class:`~torchtree.core.logger.TreeLogger` object.
        :rtype: TreeLogger
        """
        tree = process_object(data['tree_model'], dic)
        every = data.get('every', 1)
        kwargs = {}
        for key in ('file_name', 'format'):
            if key in data:
                kwargs[key] = data[key]
        return cls(tree, every, **kwargs)


@register_class
class CSV(JSONSerializable, Runnable):
    r"""
    Class for writting parameters to a CSV file.

    :param objs: list of Parameter objects
    :type objs: list[Parameter]
    """

    def __init__(self, objs: list[AbstractParameter], **kwargs) -> None:
        self.objs = objs
        self.file_name = kwargs.get('file_name', None)
        self.kwargs = kwargs

    def run(self) -> None:
        if self.file_name:
            f = open(self.file_name, 'w')
            writer = csv.writer(f, **self.kwargs)
        else:
            writer = csv.writer(sys.stdout, **self.kwargs)
        writer.writerow([obj.id for obj in self.objs])
        temp = torch.stack(list(map(lambda x: x.tensor, self.objs)))
        for i in range(temp.shape[1]):
            writer.writerow(temp[:, i].detach().numpy().tolist())
        if self.file_name:
            f.close()

    @classmethod
    def from_json(cls, data, dic) -> CSV:
        r"""
        Create a CSV object.

        :param data: json representation of CSV object.
        :type data: dict[str,Any]
        :param dic: dictionary containing additional objects that can be referenced
         in data.
        :type dic: dict[str,Any]

        :return: a :class:`~torchtree.core.logger.CSV` object.
        :rtype: CSV
        """
        params = process_objects(data['parameters'], dic)
        kwargs = {}
        for key in ('file_name', 'delimiter'):
            if key in data:
                kwargs[key] = data[key]
        return cls(params, **kwargs)


@register_class
class Dumper(JSONSerializable, Runnable):
    r"""
    Class for saving parameters to a json file.

    :param parameters: list of Parameters.
    :type parameters: list[Parameter]
    """

    def __init__(self, parameters: list[AbstractParameter], **kwargs) -> None:
        if 'file_name' in kwargs:
            self.file_name = kwargs['file_name']
            del kwargs['file_name']
        else:
            self.file_name = None
        self.kwargs = kwargs
        self.parameters = parameters

    def run(self) -> None:
        r"""
        Write the parameters to the file.
        """
        if self.file_name is not None:
            with open(self.file_name, 'w') as fp:
                json.dump(self.parameters, fp, cls=ParameterEncoder, **self.kwargs)
        else:
            json.dumps(self.parameters, cls=ParameterEncoder, **self.kwargs)

    @classmethod
    def from_json(cls, data, dic) -> Dumper:
        r"""
        Create a Dumper object.

        :param data: json representation of Dumper object.
        :type data: dict[str,Any]
        :param dic: dictionary containing additional objects that can be referenced
         in data.
        :type dic: dict[str,Any]

        :return: a :class:`~torchtree.core.logger.Dumper` object.
        :rtype: Dumper
        """
        parameters = process_objects(data['parameters'], dic)
        kwargs = {'indent': 2}
        for key in ('file_name', 'indent'):
            if key in data:
                kwargs[key] = data[key]
        return cls(parameters, **kwargs)
