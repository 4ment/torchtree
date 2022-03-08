import copy
import functools
import importlib
import json
import logging
import numbers
import os
import re
import signal
from pathlib import Path
from typing import Union

import torch

REGISTERED_CLASSES = {}


class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return {'values': obj.tolist(), 'type': str(obj.dtype)}
        return json.JSONEncoder.default(self, obj)


def as_tensor(dct, dtype=torch.float64):
    if 'type' in dct and dct['type'].startswith('torch'):
        return torch.tensor(dct['values'], dtype=dtype)
    return dct


def tensor_rand(distribution, shape, dtype=None, device=None, requires_grad=False):
    """Create a tensor with the given dtype and shape and initialize it using a
    distribution.

    Continuous distributions: normal, log_normal, uniform.
    Discrete distributions: random, bernoulli

    :param distribution: distribution as a string (e.g. 'normal(1.0,2.0)', 'normal',
     'normal()').
    :type distribution: str
    :param shape: shape of the tensor
    :type shape: Sequence[int]
    :param dtype: dtype of the tensor
    :type dtype: torch.dtype
    :param device: device of the tensor
    :type device: torch.device
    :return: tensor
    :rtype: torch.Tensor

    :example:
    >>> _ = torch.manual_seed(0)
    >>> t1 = tensor_rand('normal(1.0, 2.0)', (1,2), dtype=torch.float64)
    >>> t1
    tensor([[4.0820, 0.4131]], dtype=torch.float64)
    >>> _ = torch.manual_seed(0)
    >>> t2 = tensor_rand('normal(0.0, 1.0)', (1,2), dtype=torch.float64)
    >>> _ = torch.manual_seed(0)
    >>> t3 = tensor_rand('normal()', (1,2), dtype=torch.float64)
    >>> t2 == t3
    tensor([[True, True]])
    """

    temp = list(filter(None, re.split(r'[(),]', distribution)))
    name = temp[0]
    params = [] if len(temp) == 1 else [float(p) for p in temp[1:]]
    if name == 'normal':
        tensor = torch.empty(shape, dtype=dtype, device=device).normal_(*params)
    elif name[0] == 'log_normal':
        tensor = torch.empty(shape, dtype=dtype, device=device).log_normal_(*params)
    elif name[0] == 'uniform':
        tensor = torch.empty(shape, dtype=dtype, device=device).uniform_(*params)
    elif name[0] == 'random':
        tensor = torch.empty(shape, dtype=dtype, device=device).random_(*params)
    elif name[0] == 'bernoulli':
        tensor = torch.empty(shape, dtype=dtype, device=device).bernoulli_(*params)
    else:
        raise Exception(
            '{} is not a valid distribution to initialize tensor. input: {}'.format(
                name, distribution
            )
        )
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor


def get_class(full_name: str) -> any:
    if full_name in REGISTERED_CLASSES:
        return REGISTERED_CLASSES[full_name]

    a = full_name.split('.')
    class_name = a[-1]
    module_name = '.'.join(a[:-1])
    module = importlib.import_module(module_name)
    klass = getattr(module, class_name)
    return klass


class JSONParseError(Exception):
    ...


def process_objects(data, dic):
    if isinstance(data, list):
        return [process_object(obj, dic) for obj in data]
    else:
        return process_object(data, dic)


def process_object(data, dic):
    if isinstance(data, str):
        # for references such as branches.{0:3}
        try:
            if '{' in data:
                stem, indices = data.split('{')
                start, stop = indices.rstrip('}').split(':')
                for i in range(int(start), int(stop)):
                    obj = dic[stem + str(i)]
            else:
                obj = dic[data]
        except KeyError:
            raise JSONParseError(
                'Object with ID `{}\' not found'.format(data)
            ) from None
    elif isinstance(data, dict):
        id_ = data['id']
        if id_ in dic:
            raise JSONParseError('Object with ID `{}\' already exists'.format(id_))
        if 'type' not in data:
            raise JSONParseError(
                'Object with ID `{}\' does not have a type'.format(id_)
            )

        try:
            klass = get_class(data['type'])
        except ModuleNotFoundError as e:
            raise JSONParseError(
                str(e) + " in object with ID '" + data['id'] + "'"
            ) from None
        except AttributeError as e:
            raise JSONParseError(
                str(e) + " in object with ID '" + data['id'] + "'"
            ) from None
        except ValueError as e:
            raise JSONParseError(
                str(e) + " in object with ID '" + data['id'] + "'"
            ) from None

        obj = klass.from_json_safe(data, dic)
        dic[id_] = obj
    else:
        raise JSONParseError(
            'Object is not valid (should be str or object)\nProvided: {}'.format(data)
        )
    return obj


class SignalHandler:
    def __init__(self):
        self.stop = False
        signal.signal(signal.SIGINT, self.exit)

    def exit(self, signum, frame):
        self.stop = True
        signal.signal(signal.SIGINT, signal.default_int_handler)


def validate(data, rules):
    allowed_keys_set = {'type', 'instanceof', 'list', 'constraint', 'optional'}

    # check rules were written properly
    for rule in rules:
        assert ('type' in rules[rule]) is True
        diff = set(rules[rule].keys()).difference(allowed_keys_set)
        if len(diff) != 0:
            print('Not allowed', diff)

    # check missing keys in json
    for rule_key in rules.keys():
        if rule_key not in data and rules[rule_key].get('optional', False) is not True:
            raise ValueError('Missing key: {}'.format(rule_key))

    # check keys in json
    for datum_key in data.keys():
        if datum_key not in ('id', 'type') and datum_key not in rules:
            raise ValueError('Key not allowed: {}'.format(datum_key))

    types = dict(
        zip(
            ['string', 'bool', 'int', 'float', 'object', 'numbers.Number'],
            [str, bool, int, float, dict, numbers.Number],
        )
    )
    for datum_key in data.keys():
        if datum_key not in ('id', 'type'):
            # if not isinstance(datum_key, str):
            #     raise ValueError('{} should be a string'.format(datum_key))
            # continue

            # check type
            type_found = None
            for rule_type in rules[datum_key]['type'].split('|'):
                if rules[datum_key].get('list', False):
                    all_ok = all(
                        isinstance(x, types.get(rule_type)) for x in data[datum_key]
                    )
                    if all_ok:
                        type_found = True
                        break
                elif isinstance(data[datum_key], types.get(rule_type)):
                    type_found = True
                    break
            if type_found is None:
                raise ValueError('\'type\' is not valid: {}'.format(data[datum_key]))


def remove_comments(obj):
    if isinstance(obj, list):
        for element in obj:
            remove_comments(element)
    elif isinstance(obj, dict):
        for key in list(obj.keys()).copy():
            if not key.startswith('_'):
                remove_comments(obj[key])
            else:
                del obj[key]


def replace_wildcard_with_str(obj, wildcard, value):
    if isinstance(obj, list):
        for element in obj:
            replace_wildcard_with_str(element, wildcard, value)
    elif isinstance(obj, dict):
        for key in list(obj.keys()).copy():
            replace_wildcard_with_str(obj[key], wildcard, value)
            if key == 'id' and wildcard in obj[key]:
                obj[key] = obj[key].replace(wildcard, value)


def replace_star_with_str(obj, value):
    if isinstance(obj, list):
        for element in obj:
            replace_star_with_str(element, value)
    elif isinstance(obj, dict):
        for key in list(obj.keys()).copy():
            replace_star_with_str(obj[key], value)
            if key == 'id' and obj[key][-1] == '*':
                obj[key] = obj[key][:-1] + value


def expand_plates(obj, parent=None, idx=None):
    if isinstance(obj, list):
        for i, element in enumerate(obj):
            expand_plates(element, obj, i)
    elif isinstance(obj, dict):
        if 'type' in obj and obj['type'].endswith('Plate'):
            if 'range' in obj:
                r = list(map(int, obj['range'].split(':')))
                objects = []
                for i in range(*r):
                    clone = copy.deepcopy(obj['object'])
                    if 'var' in obj:
                        replace_wildcard_with_str(clone, f"${{{obj['var']}}}", str(i))
                    else:
                        replace_star_with_str(clone, str(i))
                    objects.append(clone)
                # replace plate dict with object list in parent list
                if idx is not None:
                    del parent[idx]
                    parent[idx:idx] = objects
                else:
                    raise JSONParseError('plate works only when part of a list')
        else:
            for value in obj.values():
                expand_plates(value, obj, None)


def update_parameters(json_object, parameters) -> None:
    """Recursively replace tensor in json_object with tensors present in
    parameters.

    :param dict json_object: json object
    :param parameters: list of Parameters
    :type parameters: list(Parameter)
    """
    if isinstance(json_object, list):
        for element in json_object:
            update_parameters(element, parameters)
    elif isinstance(json_object, dict):
        if 'type' in json_object and json_object['type'] in (
            'torchtree.core.parameter.Parameter',
            'torchtree.Parameter',
            'Parameter',
        ):
            if json_object['id'] in parameters:
                # get rid of full, full_like, tensor...
                for key in list(json_object.keys()).copy():
                    if key not in ('id', 'type', 'dtype', 'nn'):
                        del json_object[key]
                # set new tensor
                json_object['tensor'] = parameters[json_object['id']]['tensor']
        else:
            for value in json_object.values():
                update_parameters(value, parameters)


def print_graph(g: torch.Tensor, level: int = 0) -> None:
    r"""
    Print computation graph.

    :param torch.Tensor g: a tensor
    :param level: indentation level
    """
    if g is not None:
        print('*' * level * 4, g)
        for subg in g.next_functions:
            print_graph(subg[0], level + 1)


class AlternativeAttributeError(Exception):
    """Custom exception for debugging conflicts between @property and
    __getattr__

    https://stackoverflow.com/questions/36575068/attributeerrors-undesired-interaction-between-property-and-getattr
    """

    @classmethod
    def wrapper(err_type, f):
        """wraps a function to reraise an AttributeError as the alternate
        type."""

        @functools.wraps(f)
        def alt_AttrError_wrapper(*args, **kw):
            try:
                return f(*args, **kw)
            except AttributeError as e:
                new_err = err_type(e)
                new_err.__traceback__ = e.__traceback__.tb_next
                raise new_err from None

        return alt_AttrError_wrapper


def string_to_list_index(index_str) -> Union[int, slice]:
    slice_indexes_str = index_str.split(':')
    if len(slice_indexes_str) == 1:
        index = int(slice_indexes_str[0])
    else:
        # [ <first element to include> : <first element to exclude> : <step> ]
        start = None if slice_indexes_str[0] == '' else int(slice_indexes_str[0])
        end_excluded = None
        step = None
        if len(slice_indexes_str) > 1:
            if slice_indexes_str[1] != '':
                end_excluded = int(slice_indexes_str[1])
            if len(slice_indexes_str) == 3 and slice_indexes_str[2] != '':
                step = int(slice_indexes_str[2])
        index = slice(start, end_excluded, step)
    return index


def package_contents(package_name):
    import importlib.util

    spec = importlib.util.find_spec(package_name)
    # origin is None for namespace packages (no __init__.py)
    if spec is None or spec.origin is None:
        return set()

    pathname = Path(spec.origin).parent
    ret = set()
    with os.scandir(pathname) as entries:
        for entry in entries:
            if entry.name.startswith('_'):
                continue
            current = '.'.join((package_name, entry.name.partition('.')[0]))
            if entry.is_file():
                if entry.name.endswith('.py'):
                    ret.add(current)
            elif entry.is_dir():
                ret.add(current)
                ret |= package_contents(current)
    return ret


def register_class(_cls, name=None):
    logging.info('register_class: {}'.format(_cls))
    if name is not None:
        REGISTERED_CLASSES[name] = _cls
    else:
        REGISTERED_CLASSES[_cls.__name__] = _cls
    return _cls
