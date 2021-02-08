import json
import torch
import signal
import importlib


class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return {'values': obj.tolist(), 'type': str(obj.dtype)}
        return json.JSONEncoder.default(self, obj)


def as_tensor(dct, dtype=torch.float64):
    if 'type' in dct and dct['type'].startswith('torch'):
        return torch.tensor(dct['values'], dtype=dtype)
    return dct


def get_class(full_name):
    a = full_name.split('.')
    class_name = a[-1]
    module_name = '.'.join(a[:-1])
    module = importlib.import_module(module_name)
    klass = getattr(module, class_name)
    return klass


def process_objects(data, dic):
    if isinstance(data, list):
        return [process_object(obj, dic) for obj in data]
    else:
        return process_object(data, dic)


def process_object(data, dic):
    if isinstance(data, str):
        obj = dic[data]
    elif isinstance(data, dict):
        id_ = data['id']
        if id_ in dic:
            raise ValueError('Object with ID `{}\' already exists'.format(id_))
        klass = get_class(data['type'])
        obj = klass.from_json(data, dic)
        dic[id_] = obj
    else:
        id_ = data['id']
        raise ValueError('Object with ID `{}\' is not valid (should be str or object)'.format(id_))
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

    types = dict(zip(['string', 'bool', 'int', 'float', 'object'],
                     [str, bool, int, float, dict]))
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
