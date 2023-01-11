import argparse
import glob
import os
from datetime import datetime

from postprocessors.sandbox.extract_timestamps import ExtractTimestamps


def main(in_directory: str, out_directory: str, search_pattern: str):
    """
    Extract timestamps from files matching search_pattern in in_directory into timestamp files in out_directory.

    Parameters
    ----------
    in_directory: str
        Path to directory containing widebeam experiment files.
    out_directory: str
        Path to directory to save post-processed widebeam experiment files.
    search_pattern: str
        Pattern to match when finding files
    """

    for path in glob.glob(f'{in_directory}/{search_pattern}'):
        if not os.path.isfile(path):
            continue
        if path.endswith(".site"):
            input_structure = "site"
        else:
            input_structure = "array"

        filename = os.path.basename(path)
        fields = filename.split('.')
        timestamp_file = '.'.join(fields[:5] + ['timestamps'])
        timestamp_path = f'{out_directory}/{timestamp_file}'

        start = datetime.utcnow()

        if os.path.isfile(timestamp_path):
            print(f'{path} - Already done. ', end='')

        else:
            # The input file type (antennas_iq, bfiq, rawacf) doesn't matter, so just using antennas_iq as placeholder
            # We have to process it downstream so the process_record method is called, so output_type is 'bfiq'
            ExtractTimestamps(path, timestamp_path, 'antennas_iq', 'bfiq', input_structure, 'site')

        end = datetime.utcnow()
        duration = (end - start).total_seconds()
        print(f'Time: {duration:.2f} s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('in_directory', type=str, help='Directory to search for files')
    parser.add_argument('out_directory', type=str, help='Path to save output files')
    parser.add_argument('--pattern', type=str, help='Pattern to match when searching for files',
                        default='*antennas_iq*')
    args = parser.parse_args()

    main(args.in_directory, args.out_directory, args.pattern)
