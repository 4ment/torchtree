import argparse
import math
import os


def zero_or_path(arg):
    if arg == '0':
        return 0
    elif arg is not None and not os.path.exists(arg):
        raise argparse.ArgumentTypeError(
            'invalid choice (choose from 0 or a path to a text file)'
        )
    else:
        return arg


def str_or_float(arg, choices):
    """Used by argparse when the argument can be either a number or a string
    from a prespecified list of options."""
    try:
        return float(arg)
    except ValueError:
        if (isinstance(choices, tuple) and arg in choices) or choices == arg:
            return arg
        else:
            if isinstance(choices, str):
                choices = (choices,)
            message = "'" + "','".join(choices) + '"'
            raise argparse.ArgumentTypeError(
                'invalid choice (choose from a number or ' + message + ')'
            )


def list_of_float(arg, length):
    """Used by argparse when the argument should be a list of floats."""
    values = arg.split(",")
    if values != length:
        raise argparse.ArgumentTypeError(
            f"invalid choice (list should contain {length} real numbers)"
        )
    try:
        return list(map(float, values))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid choice (list should contain {length} real numbers)"
        )


def list_or_int(arg, min=1, max=math.inf):
    """Used by argparse when the argument should be a list of ints or a int."""
    values = arg.split(",")
    if len(values) == 1 and isinstance(arg, int):
        return int(arg)
    elif min <= len(values) <= max:
        try:
            return list(map(int, values))
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(
        "invalid choice (choose from an int or a list of ints)"
    )
