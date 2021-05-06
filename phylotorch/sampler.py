import csv
import sys
from typing import List

from . import Parameter
from .core.runnable import Runnable
from .core.serializable import JSONSerializable
from .core.utils import process_objects, process_object
from .distributions.distributions import DistributionModel


class Sampler(JSONSerializable, Runnable):
    def __init__(self, model: DistributionModel, objs: List[Parameter], samples: int, **kwargs):
        self.model = model
        self.objs = objs
        self.samples = samples
        if 'file_name' in kwargs:
            self.file_name = kwargs['file_name']
            del kwargs['file_name']
        self.kwargs = kwargs

    def run(self):
        if self.file_name:
            f = open(self.file_name, 'w')
            writer = csv.writer(f, **self.kwargs)
        else:
            writer = csv.writer(sys.stdout, **self.kwargs)
        header = []
        for obj in self.objs:
            header.extend(['{}.{}'.format(obj.id, i) for i in range(obj.shape[-1])])
        writer.writerow(header)

        for i in range(self.samples):
            self.model.sample()
            row = []
            for obj in self.objs:
                row.extend(obj.tensor.detach().numpy().tolist())
            writer.writerow(row)

        if self.file_name:
            f.close()

    @classmethod
    def from_json(cls, data, dic):
        model = process_object(data['model'], dic)
        params = process_objects(data['parameters'], dic)
        samples = data['samples']
        kwargs = {}
        for key in ('file_name', 'delimiter'):
            if key in data:
                kwargs[key] = data[key]
        return cls(model, params, samples, **kwargs)
