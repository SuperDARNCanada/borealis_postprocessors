import argparse
import glob
import os
from datetime import datetime

from postprocessors import borealis_to_borealis_rename, ProcessBfiq2Rawacf, ProcessAntennasIQ2Bfiq


def main(in_directory: str, out_directory: str, out_struct: str, search_pattern: str):
    """
    Postprocess all widebeam experiments from in_directory.

    Parameters
    ----------
    in_directory: str
        Path to directory containing widebeam experiment files.
    out_directory: str
        Path to directory to save post-processed widebeam experiment files.
    out_struct: str
        Final structure to save files as. Either 'site', 'array', or 'dmap'
    search_pattern: str
        Pattern to match when finding files

    Returns
    -------

    """
    averaging_method = "mean"

    for path in glob.glob(f'{in_directory}/{search_pattern}'):
        if not os.path.isfile(path):
            continue
        if path.endswith(".site"):
            input_structure = "site"
        else:
            input_structure = "array"

        filename = os.path.basename(path)

        bfiq_file = borealis_to_borealis_rename(filename, 'bfiq', 'site')
        rawacf_site = borealis_to_borealis_rename(filename, 'rawacf', 'site')
        rawacf_array = borealis_to_borealis_rename(filename, 'rawacf', 'array')

        bfiq_path = f'{out_directory}/{bfiq_file}'
        rawacf_site_path = f'{out_directory}/{rawacf_site}'
        rawacf_array_path = f'{out_directory}/{rawacf_array}'

        if out_struct == 'site':
            rawacf_out = rawacf_site_path
        else:
            rawacf_out = rawacf_array_path

        start = datetime.utcnow()

        if os.path.isfile(rawacf_site_path) or os.path.isfile(rawacf_array_path):
            # Either one is fine, we aren't picky
            print(f'{path} - Already done. ', end='')

        elif os.path.isfile(bfiq_path):
            # bfiq already exists
            print(f'{bfiq_path} -> {rawacf_out}  ', end='')
            ProcessBfiq2Rawacf(bfiq_path, rawacf_out, 'site', out_struct, averaging_method, num_processes=1)
            os.remove(bfiq_path)       # Don't want to keep this around

        else:
            print(f'{path} -> {bfiq_path} ', end='')
            ProcessAntennasIQ2Bfiq(path, bfiq_path, input_structure, 'site', num_processes=1)
            print(f'-> {os.path.basename(rawacf_out)}  ', end='')
            ProcessBfiq2Rawacf(bfiq_path, rawacf_out, 'site', out_struct, averaging_method, num_processes=1)

        end = datetime.utcnow()
        duration = (end - start).total_seconds()
        print(f'Time: {duration:.2f} s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('in_directory', type=str, help='Directory to search for files')
    parser.add_argument('out_directory', type=str, help='Path to save output files')
    parser.add_argument('--out_structure', type=str, help='Structure to save rawacf files as', default='site')
    parser.add_argument('--pattern', type=str, help='Pattern to match when searching for files',
                        default='*antennas_iq*')
    args = parser.parse_args()

    main(args.in_directory, args.out_directory, args.out_structure, args.pattern)
