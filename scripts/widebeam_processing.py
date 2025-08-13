import glob
import os
import argparse

import postprocessors
from postprocessors import ConvertFile
from postprocessors.core.antennas_iq_to_rawacf import AntennasIQ2Rawacf
from postprocessors.sandbox.chebyshev_30db import Chebyshev30dB

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', help='Directory containing files to process')
    parser.add_argument('outdir', help='Directory to store processed files in')
    parser.add_argument('--pattern', help='Pattern to search for when globbing files from indir',
                        default='*antennas_iq.h5')
    parser.add_argument('--dmap', help='Process files to DMAP?', action='store_true')
    args = parser.parse_args()
    in_directory = args.indir
    out_directory = args.outdir
    pattern = args.pattern

    if args.dmap:
        output_structure = 'dmap'
    else:
        output_structure = 'site'

    for path in glob.glob(f'{in_directory}/{pattern}'):
        if "antennas_iq" not in path:
            continue
        if not os.path.isfile(path):
            continue
        if path.endswith(".h5") or path.endswith(".site"):
            input_structure = "site"
        else:
            input_structure = "array"

        filename = os.path.basename(path)

        if args.dmap:
            outfile = postprocessors.b2sd_rename(filename, 'rawacf')
        else:
            outfile = postprocessors.b2b_rename(filename, 'rawacf', 'site')
        outpath = out_directory + '/' + outfile
        print(f'{path}')

        # Process the file to rawacf
        if not os.path.isfile(outpath):
            print(f'\t-> {outpath}')
            processor = Chebyshev30dB(path, outpath, input_structure, output_structure)
            processor.process_file(num_processes=8, keep_intermediate_files=False, tx_pattern="60deg_fov")

        # # Process the file to dmap
        # if args.dmap and not os.path.isfile(dmap_path):
        #     print(f'\t-> {dmap_path}')
        #     ConvertFile(rawacf_path, dmap_path, "rawacf", "rawacf", 'site', 'dmap')
