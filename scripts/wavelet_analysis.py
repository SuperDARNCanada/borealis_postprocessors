import argparse
import glob
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
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
        dirname = os.path.dirname(path)
        outfile = dirname + '/' + borealis_to_borealis_rename(filename, 'dtcwt', 'site')

        start = datetime.utcnow()

        # WaveletDecomposition(path, outfile, input_structure)

        with h5py.File(outfile, 'r') as f:
            recs = sorted(list(f.keys()))
            rec = f[recs[0]]

            fig, axes = plt.subplots(6, 1, figsize=(10, 8), sharey='all')
            axes[0].plot(10 * np.log10(np.abs(rec['data'][:].reshape(rec['data_dimensions'][:])[10, 10].real)))
            axes[1].plot(10 * np.log10(np.abs(rec['detail1'][10, 10, ::2].real)))
            axes[2].plot(10 * np.log10(np.abs(rec['detail2'][10, 10, ::4].real)))
            axes[3].plot(10 * np.log10(np.abs(rec['detail3'][10, 10, ::8].real)))
            axes[4].plot(10 * np.log10(np.abs(rec['detail4'][10, 10, ::16].real)))
            axes[5].plot(10 * np.log10(np.abs(rec['approx'][10, 10, ::16].real)))
            axes[0].plot(10 * np.log10(np.abs(rec['data'][:].reshape(rec['data_dimensions'][:])[10, 10].imag)))
            axes[1].plot(10 * np.log10(np.abs(rec['detail1'][10, 10, ::2].imag)))
            axes[2].plot(10 * np.log10(np.abs(rec['detail2'][10, 10, ::4].imag)))
            axes[3].plot(10 * np.log10(np.abs(rec['detail3'][10, 10, ::8].imag)))
            axes[4].plot(10 * np.log10(np.abs(rec['detail4'][10, 10, ::16].imag)))
            axes[5].plot(10 * np.log10(np.abs(rec['approx'][10, 10, ::16].imag)))
            axes[0].plot(10 * np.log10(np.abs(rec['data'][:].reshape(rec['data_dimensions'][:])[10, 10])))
            axes[1].plot(10 * np.log10(np.abs(rec['detail1'][10, 10, ::2])))
            axes[2].plot(10 * np.log10(np.abs(rec['detail2'][10, 10, ::4])))
            axes[3].plot(10 * np.log10(np.abs(rec['detail3'][10, 10, ::8])))
            axes[4].plot(10 * np.log10(np.abs(rec['detail4'][10, 10, ::16])))
            axes[5].plot(10 * np.log10(np.abs(rec['approx'][10, 10, ::16])))
            for ax in axes:
                ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
            plt.show()
        end = datetime.utcnow()
        duration = (end - start).total_seconds()
        print(f'Time: {duration:.2f} s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=str, help='Input file')
    args = parser.parse_args()

    main(args.infile)
