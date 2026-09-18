import argparse


def str2bool(v):
    """Parse a boolean value from a string.

    Accepts common truthy/falsey strings. Also handles bool values directly and
    treats an explicit None as True for argparse flag use with nargs='?'.
    """
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    val = str(v).lower()
    if val in ("yes", "true", "t", "y", "1"):
        return True
    if val in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")
