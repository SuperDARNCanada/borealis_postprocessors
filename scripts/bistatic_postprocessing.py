import argparse
import glob
import os
import h5py
import deepdish as dd
from datetime import datetime

from postprocessors import ConvertFile, borealis_to_borealis_rename
from postprocessors.sandbox.widebeam_antennas_iq_to_bfiq import ProcessWidebeamAntennasIQ2Bfiq
from postprocessors.sandbox.bistatic_processing import BistaticProcessing


def main(in_directory: str, out_directory: str, out_struct: str, search_pattern: str, timestamp_dir: str):
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
    timestamp_dir: str
        Path to directory to search for timestamp files.

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

        bfiq_file = borealis_to_borealis_rename(filename, 'bfiq', out_struct)
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
            ConvertFile(bfiq_path, rawacf_out, 'bfiq', 'rawacf', 'site', out_struct, averaging_method)
            os.remove(bfiq_path)       # Don't want to keep this around

        else:
            # Figure out how to process the file. Some experiments are straightforward, some need some love
            if input_structure == 'site':
                with h5py.File(path, 'r') as f:
                    records = sorted(list(f.keys()))
                first_record = dd.io.load(path, f'/{records[0]}')
                experiment_name = first_record['experiment_name']
                experiment_comment = first_record['experiment_comment']
            else:
                with h5py.File(path, 'r') as f:
                    attributes = f.attrs
                    experiment_name = attributes['experiment_name']
                    experiment_comment = attributes['experiment_comment']

            if experiment_name in ['Widebeam_2tx', 'Widebeam_3tx', 'MultifreqWidebeam']:    # These ones need some love
                print(f'{path} -> {rawacf_out}  ', end='')
                ProcessWidebeamAntennasIQ2Bfiq(path, bfiq_path, input_structure, 'site')
                ConvertFile(bfiq_path, rawacf_out, 'bfiq', 'rawacf', 'site', out_struct, averaging_method)
                os.remove(bfiq_path)    # We don't need to keep these around

            elif experiment_name == 'BistaticTest':
                if 'Bistatic widebeam mode' in experiment_comment:
                    print(f'{path} - Bistatic listening experiment. ', end='')
                    # Search for file with timestamps to match against, then process.
                    fields = filename.split('.')
                    timestamp_pattern = f'{fields[0]}.{fields[1][0:2]}*timestamps*'   # YYYYMMDD.HH*timestamps*
                    timestamp_files = glob.glob(f'{timestamp_dir}/{timestamp_pattern}')
                    if len(timestamp_files) == 0:
                        print('Error - no timestamp file.  ', end='')
                        continue
                    else:   # Expect 1 file
                        print(f'Using timestamps from {timestamp_files[0]}.  ', end='')
                        BistaticProcessing(path, rawacf_out, input_structure, out_struct, timestamp_files[0])
                else:
                    print(f'{path} -> {rawacf_out}  ', end='')
                    ConvertFile(path, rawacf_out, 'antennas_iq', 'rawacf', input_structure, out_struct, averaging_method)
            else:
                print(f'{path} - Not a widebeam experiment.  ', end='')

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
    parser.add_argument('--timestamp_dir', type=str, help='Path to directory with timestamp files')
    args = parser.parse_args()

    main(args.in_directory, args.out_directory, args.out_structure, args.pattern, args.timestamp_dir)
