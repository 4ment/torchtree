import importlib
import pkgutil


class PluginManager:
    def __init__(self) -> None:
        self._plugins = {}
        self._loaded_plugins = {}

    def load_plugins(self):
        for finder, name, ispkg in pkgutil.iter_modules():
            if name.startswith('torchtree_'):
                try:
                    m = importlib.import_module(name)
                    self._plugins[name] = m.__plugin__

                    full_path = "{}.{}".format(name, m.__plugin__)
                    tmp = full_path.split('.')
                    module = importlib.import_module(".".join(tmp[:-1]))
                    class_name = tmp[-1]
                    self._loaded_plugins[m.__plugin__] = getattr(module, class_name)()
                except AttributeError:
                    pass

    def disable_plugins(self, plugins):
        for plugin in plugins:
            del self._loaded_plugins[self._plugins[plugin]]

    def load_arguments(self, subparsers):
        for plugin in self._loaded_plugins.values():
            plugin.load_arguments(subparsers)

    def plugins(self):
        return self._loaded_plugins.values()
