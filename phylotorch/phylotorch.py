import argparse
import json
import logging

import torch

from phylotorch.core.runnable import Runnable
from phylotorch.core.utils import (
    JSONParseError,
    expand_plates,
    process_objects,
    remove_comments,
    update_parameters,
)


def main():
    """Main function to run phylotorch."""
    parser = argparse.ArgumentParser(
        prog='phylotorch', description='Phylogenetic inference using pytorch'
    )
    parser.add_argument(
        'file', metavar='input-file-name', help='JSON configuration file'
    )
    parser.add_argument(
        '-c', '--checkpoint', required=False, default=None, help='JSON checkpoint file'
    )
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        required=False,
        default=None,
        help="""initialize seed""",
    )
    arg = parser.parse_args()

    if arg.seed is not None:
        torch.manual_seed(arg.seed)
    print('SEED: {}'.format(torch.initial_seed()))
    print()

    logging.basicConfig(format='%(levelname)s: %(message)s')
    with open(arg.file) as file_pointer:
        # data = json.load(fp, object_pairs_hook=collections.OrderedDict)
        data = json.load(file_pointer)

    remove_comments(data)
    expand_plates(data)

    if arg.checkpoint is not None:
        with open(arg.checkpoint) as file_pointer:
            checkpoint = json.load(file_pointer)
            tensors = {}
            for param in checkpoint:
                tensors[param['id']] = param
            update_parameters(data, tensors)

    dic = {}
    try:
        for element in data:
            obj = process_objects(element, dic)
            if isinstance(obj, Runnable):
                obj.run()
    except JSONParseError as error:
        logging.error(error)


if __name__ == "__main__":
    main()
