import argparse
import json

from torchtree._version import __version__
from torchtree.cli import PLUGIN_MANAGER
from torchtree.cli.advi import create_variational_parser
from torchtree.cli.hmc import create_hmc_parser
from torchtree.cli.map import create_map_parser
from torchtree.cli.mcmc import create_mcmc_parser


def main():
    parser = argparse.ArgumentParser(
        prog='torchtree-cli',
        description='Command line interface for creating JSON file for torchtree',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subprasers = parser.add_subparsers()

    create_variational_parser(subprasers)

    create_map_parser(subprasers)

    create_mcmc_parser(subprasers)

    create_hmc_parser(subprasers)

    PLUGIN_MANAGER.load_plugins()
    PLUGIN_MANAGER.load_arguments(subprasers)

    arg = parser.parse_args()
    if not hasattr(arg, "func"):
        parser.print_help()
        exit(2)

    json_dic = arg.func(arg)

    print(json.dumps(json_dic, indent=2))
