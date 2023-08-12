import argparse
import glob
import os
import h5py
import deepdish as dd
from datetime import datetime

from postprocessors import ConvertFile, borealis_to_borealis_rename
from postprocessors.sandbox.wavelet_decomp import WaveletDecomposition


def main(infile: str):
    """
    Postprocess all widebeam experiments from in_directory.

    Parameters
    ----------
    infile: str
        Path to file.

    Returns
    -------

    """
    for path in glob.glob(f'{infile}'):
        if not os.path.isfile(path):
            continue
        if path.endswith(".site"):
            input_structure = "site"
        else:
            input_structure = "array"

        filename = os.path.basename(path)

        outfile = borealis_to_borealis_rename(filename, 'wavelet', 'site')

        start = datetime.utcnow()

        WaveletDecomposition(path, outfile, input_structure)

        end = datetime.utcnow()
        duration = (end - start).total_seconds()
        print(f'Time: {duration:.2f} s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=str, help='Input file')
    args = parser.parse_args()

    main(args.infile)
