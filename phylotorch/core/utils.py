import importlib
import json
import numbers
import re
import signal

import torch


class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return {'values': obj.tolist(), 'type': str(obj.dtype)}
        return json.JSONEncoder.default(self, obj)


def as_tensor(dct, dtype=torch.float64):
    if 'type' in dct and dct['type'].startswith('torch'):
        return torch.tensor(dct['values'], dtype=dtype)
    return dct


def tensor_rand(distribution, dtype, shape):
    """ Create a tensor with the given dtype and shape and initialize it using a distribution.

    Continuous distributions: normal, log_normal, uniform.
    Discrete distributions: random, bernoulli

    :param distribution: distribution as a string (e.g. 'normal(1.0,2.0)', 'normal', 'normal()').
    :type distribution: str
    :param dtype: dtype of the tensor
    :type dtype: torch.dtype
    :param shape: shape of the tensor
    :type shape: Sequence[int]
    :return: tensor
    :rtype: torch.Tensor

    :example:
    >>> _ = torch.manual_seed(0)
    >>> t1 = tensor_rand('normal(1.0, 2.0)', torch.float64, (1,2))
    >>> t1
    tensor([[4.0820, 0.4131]], dtype=torch.float64)
    >>> _ = torch.manual_seed(0)
    >>> t2 = tensor_rand('normal(0.0, 1.0)', torch.float64, (1,2))
    >>> _ = torch.manual_seed(0)
    >>> t3 = tensor_rand('normal()', torch.float64, (1,2))
    >>> t2 == t3
    tensor([[True, True]])
    """

    temp = list(filter(None, re.split(r'[(),]', distribution)))
    name = temp[0]
    params = [] if len(temp) == 1 else [float(p) for p in temp[1:]]
    if name == 'normal':
        tensor = torch.empty(shape, dtype=dtype).normal_(*params)
    elif name[0] == 'log_normal':
        tensor = torch.empty(shape, dtype=dtype).log_normal_(*params)
    elif name[0] == 'uniform':
        tensor = torch.empty(shape, dtype=dtype).uniform_(*params)
    elif name[0] == 'random':
        tensor = torch.empty(shape, dtype=dtype).random_(*params)
    elif name[0] == 'bernoulli':
        tensor = torch.empty(shape, dtype=dtype).bernoulli_(*params)
    else:
        raise Exception('{} is not a valid distribution to initialize tensor. input: {}'.format(name, distribution))
    return tensor


def get_class(full_name):
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
        try:
            obj = dic[data]
        except KeyError:
            raise JSONParseError('Object with ID `{}\' not found'.format(data))
    elif isinstance(data, dict):
        id_ = data['id']
        if id_ in dic:
            raise JSONParseError('Object with ID `{}\' already exists'.format(id_))
        if 'type' not in data:
            raise JSONParseError('Object with ID `{}\' does not have a type'.format(id_))

        try:
            klass = get_class(data['type'])
        except ModuleNotFoundError as e:
            raise JSONParseError(str(e) + " in object with ID '" + data['id'] + "'")
        except AttributeError as e:
            raise JSONParseError(str(e) + " in object with ID '" + data['id'] + "'")

        obj = klass.from_json_safe(data, dic)
        dic[id_] = obj
    else:
        raise JSONParseError('Object is not valid (should be str or object)')
    return obj


class SignalHandler():
    def __init__(self):
        self.stop = False
        signal.signal(signal.SIGINT, self.exit)

    def exit(self, signum, frame):
        self.stop = True


def validate(data, rules):
    data_keys_set = set(data.keys())
    rules_keys_set = set(rules.keys())
    allowed_keys_set = {'type', 'instanceof', 'list', 'constraint', 'optional'}

    # check rules were written properly
    for rule in rules:
        assert ('type' in rules[rule]) is True
        diff = set(rules[rule].keys()).difference(allowed_keys_set)
        if len(diff) != 0:
            print('Not allowed', diff)

    # data_keys_set.difference(rules_keys_set)
    # check missing keys in json
    for rule_key in rules.keys():
        if rule_key not in data and rules[rule_key].get('optional', False) is not True:
            raise ValueError('Missing key: {}'.format(rule_key))

    # check keys in json
    for datum_key in data.keys():
        if datum_key not in ('id', 'type') and datum_key not in rules:
            raise ValueError('Key not allowed: {}'.format(datum_key))

    types = dict(zip(['string', 'bool', 'int', 'float', 'object', 'numbers.Number'],
                     [str, bool, int, float, dict, numbers.Number]))
    for datum_key in data.keys():
        if datum_key not in ('id', 'type'):
            # if not isinstance(datum_key, str):
            #     raise ValueError('{} should be a string'.format(datum_key))
            # continue

            # check type
            type = None
            for rule_type in rules[datum_key]['type'].split('|'):
                if rules[datum_key].get('list', False):
                    all_ok = all(isinstance(x, types.get(rule_type)) for x in data[datum_key])
                    if all_ok:
                        type = True
                        break
                elif isinstance(data[datum_key], types.get(rule_type)):
                    type = True
                    break
            if type is None:
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
