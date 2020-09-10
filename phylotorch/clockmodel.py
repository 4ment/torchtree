from abc import abstractmethod

from phylotorch.model import Model


class ClockModel(Model):
    @abstractmethod
    def rates(self):
        pass


class SimpleClockModel(Model):
    def __init__(self, rates):
        self._rates_key, self._rates = rates

    def rates(self):
        return self._rates

    def update(self, value):
        if isinstance(value, dict):
            if self._rates_key in value:
                self._rates = value[self._rates_key]
        else:
            self._rates = value
