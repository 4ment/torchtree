import argparse
import json
import logging

import torch

from phylotorch.core.runnable import Runnable
from phylotorch.core.utils import JSONParseError, remove_comments, expand_plates
from phylotorch.core.utils import process_objects, update_parameters


def main():
    parser = argparse.ArgumentParser(prog='phylotorch', description='Phylogenetic inference using pytorch')
    parser.add_argument('file', metavar='input-file-name', help='JSON configuration file')
    parser.add_argument('--checkpoint', required=False, default=None, help='JSON checkpoint file')
    parser.add_argument('--seed', type=int, required=False, default=None, help="""initialize seed""")
    arg = parser.parse_args()

    if arg.seed is not None:
        torch.manual_seed(arg.seed)
    print('SEED: {}'.format(torch.initial_seed()))
    print()

    logging.basicConfig(format='%(levelname)s: %(message)s')
    with open(arg.file) as fp:
        # data = json.load(fp, object_pairs_hook=collections.OrderedDict)
        data = json.load(fp)

    remove_comments(data)
    expand_plates(data)

    if arg.checkpoint is not None:
        with open(arg.checkpoint) as fp:
            checkpoint = json.load(fp)
            tensors = {}
            for param in checkpoint:
                tensors[param['id']] = param
            update_parameters(data, tensors)

    dic = {}
    try:
        for d in data:
            obj = process_objects(d, dic)
            if isinstance(obj, Runnable):
                obj.run()
    except JSONParseError as e:
        logging.error(e)


if __name__ == "__main__":
    main()
