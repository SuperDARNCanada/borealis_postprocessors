import glob
import os
import argparse

import postprocessors
from postprocessors.sandbox.near_range import NearRange

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', help='Directory containing files to process')
    parser.add_argument('outdir', help='Directory to store processed files in')
    parser.add_argument('--pattern', help='Pattern to search for when globbing files from indir',
                        default='*antennas_iq.h5')
    args = parser.parse_args()
    in_directory = args.indir
    out_directory = args.outdir
    pattern = args.pattern

    output_structure = 'dmap'

    for path in glob.glob(f'{in_directory}/{pattern}'):
        if "antennas_iq" not in path:
            continue
        if not os.path.isfile(path):
            continue
        if path.endswith(".h5") or path.endswith(".site"):
            input_structure = "site"
        else:
            continue
            # input_structure = "array"

        filename = os.path.basename(path)

        outfile = postprocessors.b2sd_rename(filename, 'rawacf')
        outpath = out_directory + '/' + outfile
        print(f'{path}')

        # Process the file to rawacf
        if not os.path.isfile(outpath):
            print(f'\t-> {outpath}')
            processor = NearRange(path, outpath, input_structure, "dmap")
            processor.process_file(num_processes=8, keep_intermediate_files=False)
