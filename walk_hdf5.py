import h5py
import argparse
import deepdish as dd


def usage_msg():
    """
    Return the usage message for this process.

    This is used if a -h flag or invalid arguments are provided.
    """

    usage_message = """ walk_hdf5.py [-h] filename

    A more elegant way to dump and hdf5 file than H5Dump."""

    return usage_message


def script_parser():
    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("filename", help="Path to the file to be walked.")

    return parser


def walk_hdf5(filepath):
    """
    Walks the tree for a given hdf5 file.

    Parameters
    ----------
        filepath : string
            File path and name to hdf5 file to be walked.

    Returns
    -------
        None
    """
    recs = dd.io.load(filepath)
    sorted_groups = sorted(list(recs.keys()))
    print(filepath)
    for group_num, group_name in enumerate(sorted_groups):
        sorted_keys = sorted(list(recs[group_name].keys()))
        print(group_name)
        for key_num, key_name in enumerate(sorted_keys):
            print(f"        {key_name}:", recs[group_name][key_name])

    #f = h5py.File(filepath, 'r')
    #f.visititems(_print_attrs)
    return None


def _print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print("    %s: %s" % (key, val))
    return None


if __name__ == '__main__':
    parser = script_parser()
    args = parser.parse_args()
    walk_hdf5(args.filename)
