import argparse
import json

from torchtree.cli.advi import create_variational_parser
from torchtree.cli.hmc import create_hmc_parser
from torchtree.cli.map import create_map_parser


def main():
    parser = argparse.ArgumentParser(
        prog='torchtree-cli',
        description='Command line interface for creating JSON file for torchtree',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subprasers = parser.add_subparsers()

    create_variational_parser(subprasers)

    create_map_parser(subprasers)

    create_hmc_parser(subprasers)

    arg = parser.parse_args()
    json_dic = arg.func(arg)

    print(json.dumps(json_dic, indent=2))
