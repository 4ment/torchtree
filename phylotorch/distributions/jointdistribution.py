from collections import OrderedDict
from inspect import signature

import torch.distributions


class JointDistribution(torch.distributions.Distribution):

    def __init__(self, distributions, parameters=None, transforms=None):
        """ Instantiate a JointDistribution

        :param distributions: `collections.OrderedDict` of distributions
        :param parameters: `dict` or `collections.OrderedDict` of parameters
        :param transforms: `collections.OrderedDict` of transforms
        """
        super(JointDistribution, self).__init__()
        self.distributions = OrderedDict()
        self._parameters = {} if parameters is None else parameters
        self._transforms = OrderedDict()
        self.transformed_params = {}
        if transforms is not None:
            for name, fn in transforms.items():
                self.add_transform(name, fn)
        for name, dist in distributions.items():
            self.add_distribution(name, dist)
        self._joint_log_prob = 0

    @property
    def parameters(self):
        return list(self._parameters.values())

    @property
    def transforms(self):
        return self._transforms

    def add_distribution(self, name, distribution):
        self.distributions[name] = distribution

    def add_transform(self, name, fn):
        self._transforms[name] = fn

    def rsample(self, sample_shape=torch.Size()):
        samples = {}
        for name, dist in self.distributions.items():
            # it is a lambda or function
            if callable(dist):
                params = []
                for p in signature(dist).parameters.keys():
                    if p in self._parameters:
                        params.append(self._parameters[p])
                samples[name] = dist(*params).rsample(sample_shape)
            # it is a distribution
            else:
                samples[name] = dist.rsample(sample_shape=sample_shape)
        return samples

    def apply_transform(self, name, value):
        params = []
        for p in signature(self._transforms[name]).parameters.keys():
            if p in self.transformed_params:
                params.append(self.transformed_params[p])
            elif p in value:
                params.append(value[p])
            else:
                self.apply_transform(p, value)
                params.append(self.transformed_params[p])
        transform_evaluated = self._transforms[name](*params)
        if callable(transform_evaluated):
            self.transformed_params[name] = transform_evaluated(*params)
            if hasattr(transform_evaluated, 'log_abs_det_jacobian'):
                self._joint_log_prob += transform_evaluated.log_abs_det_jacobian(*params, self.transformed_params[name]).sum()
        else:
            self.transformed_params[name] = transform_evaluated

    def log_prob(self, value):
        self._joint_log_prob = 0.0
        self.transformed_params = {}
        for name, dist in self.distributions.items():
            if callable(dist):
                params = []
                # get arguments of lambda
                for p in signature(dist).parameters.keys():
                    if p in value:
                        params.append(value[p])
                    elif p in self._parameters:
                        params.append(self._parameters[p])
                    # argument is a transformed parameter and has already been parsed
                    elif p in self.transformed_params:
                        params.append(self.transformed_params[p])
                    # argument is a transformed parameter and needs to be been parsed
                    else:
                        self.apply_transform(p, value)
                        params.append(self.transformed_params[p])

                if name in value:
                    self._joint_log_prob += dist(*params).log_prob(value[name]).sum()
                else:
                    if name not in self.transformed_params:
                        self.apply_transform(name, value)
                    self._joint_log_prob += dist(*params).log_prob(self.transformed_params[name]).sum()
            else:
                if name in value:
                    self._joint_log_prob += dist.log_prob(value[name]).sum()
                else:
                    if name not in self.transformed_params:
                        self.apply_transform(name, value)
                    self._joint_log_prob += dist.log_prob(self.transformed_params[name]).sum()
        return self._joint_log_prob

    def __str__(self):
        model_string = ''
        # for name, p in self._parameters.items():
        #     model_string += name + ' ' + str(type(p)) + '\n'
        for name, transform in self._transforms.items():
            if len(signature(transform).parameters) != 0:
                model_string += name + ' := ' + transform.__name__ + ' ' + ', '.join(
                    signature(transform).parameters.keys()) + '\n'
        model_string += '\n'

        for name, dist in self.distributions.items():
            model_string += name + ' ~ '
            # it is a lambda or function
            if callable(dist):
                params = []
                for p in signature(dist).parameters.keys():
                    if p in self._parameters:
                        params.append(self._parameters[p])
                try:
                    model_string += dist(*params).__class__.__name__
                except:
                    pass
                model_string += '(' + ', '.join(signature(dist).parameters.keys()) + ')'
            # it is a distribution
            else:
                model_string += str(dist)
            model_string += '\n'
        return model_string


class InvertLikelihood(object):
    def __init__(self, like, *args):
        self.like = like
        self.args = args

    def log_prob(self, ignore):
        return self.like.log_prob(*self.args)
