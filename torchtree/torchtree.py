import argparse
import importlib
import json
import logging
import sys

import torch

from .core.runnable import Runnable
from .core.utils import (
    JSONParseError,
    TensorDecoder,
    expand_plates,
    get_class,
    package_contents,
    process_objects,
    remove_comments,
    update_parameters,
)


def main():
    """Main function to run torchtree."""
    parser = argparse.ArgumentParser(
        prog='torchtree', description='Phylogenetic inference using pytorch'
    )
    parser.add_argument(
        'file',
        type=argparse.FileType('r'),
        metavar='input-file-name',
        default=sys.stdin,
        help='JSON configuration file',
    )
    parser.add_argument(
        '-c', '--checkpoint', action='append', help='JSON checkpoint file'
    )
    parser.add_argument(
        '--dry',
        action='store_true',
        help='do not run anything, just parse',
    )
    parser.add_argument(
        '--dtype',
        required=False,
        choices=['float32', 'float64'],
        default='float64',
        help='``torch.Tensor`` type to floating point tensor type (default: float64)',
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

    logging.basicConfig(format='%(levelname)s: %(message)s')

    if arg.seed is not None:
        torch.manual_seed(arg.seed)
    print('SEED: {}'.format(torch.initial_seed()))

    dtype_class = get_class('torch.' + arg.dtype)
    torch.set_default_dtype(dtype_class)
    print('dtype: {}'.format('torch.' + arg.dtype))

    print()

    # register classes that do not require module specification
    for module in package_contents('torchtree'):
        importlib.import_module(module)

    data = json.load(arg.file)

    remove_comments(data)
    expand_plates(data)

    others = {}
    # update the parameters of the models first
    if arg.checkpoint is not None:
        for checkpoint_file in arg.checkpoint:
            with open(checkpoint_file) as file_pointer:
                checkpoint = json.load(file_pointer, cls=TensorDecoder)
                tensors = {}
                for param in checkpoint:
                    if param["type"] in ("torchtree.Parameter", "Parameter"):
                        tensors[param['id']] = param
                    else:
                        others[param['id']] = param
                update_parameters(data, tensors)
    dic = {}
    try:
        for element in data:
            obj = process_objects(element, dic)
            # now we update the state_dict of the algorithms (e.g. Optimizer, MCMC)
            if (
                arg.checkpoint is not None
                and hasattr(obj, "id")
                and obj.id in others
                and hasattr(obj, "load_state_dict")
            ):
                obj.load_state_dict(others[obj.id])

            if isinstance(obj, Runnable) and not arg.dry:
                obj.run()
    except JSONParseError as error:
        logging.error(error)


if __name__ == "__main__":
    main()
