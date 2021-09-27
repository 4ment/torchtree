import argparse
import json

from phylotorch.cli.advi import create_variational_parser
from phylotorch.cli.map import create_map_parser


def main():
    parser = argparse.ArgumentParser(
        prog='phylotorch-cli',
        description='Command line interface for creating JSON file for phylotorch',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subprasers = parser.add_subparsers()

    create_variational_parser(subprasers)

    create_map_parser(subprasers)

    arg = parser.parse_args()
    json_dic = arg.func(arg)

    print(json.dumps(json_dic, indent=2))
