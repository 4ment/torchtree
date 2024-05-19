from __future__ import annotations

import argparse
import importlib
import json

from torchtree._version import __version__
from torchtree.cli import PLUGIN_MANAGER, evolution
from torchtree.cli.advi import create_variational_parser
from torchtree.cli.hmc import create_hmc_parser
from torchtree.cli.map import create_map_parser
from torchtree.cli.mcmc import create_mcmc_parser
from torchtree.cli.utils import remove_constraints
from torchtree.core.utils import package_contents


def create_show_parser(subprasers):
    parser = subprasers.add_parser("show", help="Show some information")
    parser.add_argument('what')
    from torchtree.core.utils import REGISTERED_CLASSES

    def show(arg):
        if arg.what == "classes":
            for module in package_contents('torchtree'):
                importlib.import_module(module)
            for klass in REGISTERED_CLASSES:
                print(klass)
        elif arg.what == "plugins":
            for plugin in PLUGIN_MANAGER.plugins():
                print(plugin)
        exit(0)

    parser.set_defaults(func=show)


def main():
    parser = argparse.ArgumentParser(
        prog='torchtree-cli',
        description='Command line interface for creating JSON file for torchtree',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument('--debug', action="store_true", help=argparse.SUPPRESS)

    subprasers = parser.add_subparsers()

    create_variational_parser(subprasers)

    create_map_parser(subprasers)

    create_mcmc_parser(subprasers)

    create_hmc_parser(subprasers)

    create_show_parser(subprasers)

    PLUGIN_MANAGER.load_plugins()
    PLUGIN_MANAGER.load_arguments(subprasers)

    arg = parser.parse_args()
    if not hasattr(arg, "func"):
        parser.print_help()
        exit(2)

    evolution.check_arguments(arg, parser)

    json_dic = arg.func(arg)

    if not arg.debug:
        remove_constraints(json_dic)

    print(json.dumps(json_dic, indent=2))
