from abc import ABC, abstractmethod


class Plugin(ABC):
    @abstractmethod
    def load_arguments(self, subparsers):
        ...

    def process_tree_likelihood(self, arg, data):
        pass

    def process_coalescent(self, arg, data):
        pass

    def process_all(self, arg, data):
        pass
