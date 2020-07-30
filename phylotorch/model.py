import abc


class Model(abc.ABC):
    @abc.abstractmethod
    def update(self, value):
        pass
