from __future__ import annotations

import csv
import re
from typing import Union

import torch


def convert_date_to_real(day, month, year):
    if year % 4 == 0:
        days = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    else:
        days = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

    for i in range(month - 1):
        day += days[i]

    return (day - 1) / sum(days) + year


def read_dates_from_csv(input_file, date_format=None):
    dates = {}
    with open(input_file) as fp:
        reader = csv.reader(
            fp,
            quotechar='"',
            delimiter=',',
            quoting=csv.QUOTE_ALL,
            skipinitialspace=True,
        )
        for line in reader:
            index_name = line.index('strain')
            index_date = line.index('date')
            break
        for line in reader:
            dates[line[index_name]] = line[index_date]

    if date_format is not None:
        res = re.split(r"[/-]", date_format)
        yy = res.index('yyyy')
        MM = res.index('MM')
        dd = res.index('dd')

        for key, date in dates.items():
            res1 = re.split(r"[/-]", date)
            dates[key] = convert_date_to_real(
                int(res1[dd]), int(res1[MM]), int(res1[yy])
            )
    return dates


def make_unconstrained(json_object: Union[dict, list]) -> tuple[list[str], list[dict]]:
    """Returns a list of constrained parameter IDs (str) with the corresponding
    parameters (dict)"""
    parameters = []
    parameters_unres = []
    if isinstance(json_object, list):
        for element in json_object:
            params_unres, params = make_unconstrained(element)
            parameters_unres.extend(params_unres)
            parameters.extend(params)
    elif isinstance(json_object, dict):
        if 'type' in json_object and json_object['type'] == 'Parameter':
            if 'lower' in json_object and 'upper' in json_object:
                if json_object['lower'] == 0 and json_object['upper'] == 1:
                    json_object['type'] = 'TransformedParameter'
                    json_object['transform'] = 'torch.distributions.SigmoidTransform'
                    json_object['x'] = {
                        'id': json_object['id'] + '.unres',
                        'type': 'Parameter',
                    }
                    transform = torch.distributions.SigmoidTransform()
                    if 'tensor' in json_object and isinstance(
                        json_object['tensor'], list
                    ):
                        json_object['x']['tensor'] = transform.inv(
                            torch.tensor(json_object['tensor'])
                        ).tolist()
                    elif 'full' in json_object:
                        json_object['x']['tensor'] = transform.inv(
                            torch.tensor(json_object['tensor'])
                        ).item()
                        json_object['x']['full'] = json_object['full']
                        del json_object['full']
                    elif 'full_like' in json_object:
                        json_object['x']['tensor'] = transform.inv(
                            torch.tensor(json_object['tensor'])
                        ).item()
                        json_object['x']['full_like'] = json_object['full_like']
                        del json_object['full_like']
                    del json_object['tensor']
                elif json_object['lower'] != json_object['upper']:
                    raise NotImplementedError

                parameters.append(json_object['id'])
                parameters_unres.append(json_object['x'])
            elif 'lower' in json_object:
                if json_object['lower'] > 0:
                    json_object['type'] = 'TransformedParameter'
                    json_object['transform'] = 'torch.distributions.AffineTransform'
                    json_object['parameters'] = {
                        'loc': json_object['lower'],
                        'scale': 1.0,
                    }
                    transform = torch.distributions.AffineTransform(
                        json_object['lower'], 1.0
                    )
                    new_value = (
                        transform.inv(torch.tensor(json_object['tensor']))
                    ).tolist()

                    json_object['x'] = {
                        'id': json_object['id'] + '.unshifted',
                        'type': 'Parameter',
                        'tensor': new_value,
                        'lower': 0.0,
                    }
                    del json_object['tensor']

                    params_unres, params = make_unconstrained(json_object['x'])
                    parameters_unres.extend(params_unres)
                    parameters.extend(params)
                else:
                    json_object['type'] = 'TransformedParameter'
                    json_object['transform'] = 'torch.distributions.ExpTransform'

                    json_object['x'] = {
                        'id': json_object['id'] + '.unres',
                        'type': 'Parameter',
                    }
                    transform = torch.distributions.ExpTransform()

                    if 'full' in json_object:
                        json_object['x']['tensor'] = transform.inv(
                            torch.tensor(json_object['tensor'])
                        ).item()
                        json_object['x']['full'] = json_object['full']
                        del json_object['full']
                    elif 'full_like' in json_object:
                        json_object['x']['tensor'] = transform.inv(
                            torch.tensor(json_object['tensor'])
                        ).item()
                        json_object['x']['full_like'] = json_object['full_like']
                        del json_object['full_like']
                    else:
                        json_object['x']['tensor'] = transform.inv(
                            torch.tensor(json_object['tensor'])
                        ).tolist()

                    del json_object['tensor']

                    parameters.append(json_object['id'])
                    parameters_unres.append(json_object['x'])
            elif 'simplex' in json_object and json_object['simplex']:
                json_object['type'] = 'TransformedParameter'
                json_object['transform'] = 'torch.distributions.StickBreakingTransform'
                transform = torch.distributions.StickBreakingTransform()
                if 'full' in json_object:
                    tensor_unres = transform.inv(
                        torch.full(json_object['full'], json_object['tensor'])
                    ).tolist()
                else:
                    tensor_unres = transform.inv(
                        torch.tensor(json_object['tensor'])
                    ).tolist()

                json_object['x'] = {
                    'id': json_object['id'] + '.unres',
                    'type': 'Parameter',
                    'tensor': tensor_unres,
                }
                del json_object['tensor']
                if 'full' in json_object:
                    del json_object['full']

                parameters.append(json_object['id'])
                parameters_unres.append(json_object['x'])
            else:
                parameters.append(json_object['id'])
                parameters_unres.append(json_object)

        else:
            for element in json_object.values():
                params_unres, params = make_unconstrained(element)
                parameters.extend(params)
                parameters_unres.extend(params_unres)
    return parameters_unres, parameters
