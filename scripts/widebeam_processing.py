import glob
import os
import argparse

import postprocessors
from postprocessors import ConvertFile
from postprocessors.sandbox.hamming_beams import HammingWindowBeamforming

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', help='Directory containing files to process')
    parser.add_argument('outdir', help='Directory to store processed files in')
    parser.add_argument('--pattern', help='Pattern to search for when globbing files from indir',
                        default='*antennas_iq.hdf5.site')
    parser.add_argument('--dmap', help='Process files to DMAP?', action='store_true')
    args = parser.parse_args()
    in_directory = args.indir
    out_directory = args.outdir
    pattern = args.pattern

    output_structure = 'site'

    for path in glob.glob(f'{in_directory}/{pattern}'):
        if "antennas_iq" not in path:
            continue
        if not os.path.isfile(path):
            continue
        if path.endswith(".site"):
            input_structure = "site"
        else:
            input_structure = "array"

        filename = os.path.basename(path)

        rawacf_file = postprocessors.borealis_to_borealis_rename(filename, 'rawacf', 'site')
        dmap_file = postprocessors.borealis_to_sdarn_rename(filename, 'rawacf')
        rawacf_path = out_directory + '/' + rawacf_file
        dmap_path = out_directory + '/' + dmap_file
        print(f'{path}')

        # Process the file to rawacf
        if not os.path.isfile(rawacf_path):
            print(f'\t-> {rawacf_path}')
            HammingWindowBeamforming(path, rawacf_path, input_structure, output_structure, num_processes=5)

        # Process the file to dmap
        if args.dmap and not os.path.isfile(dmap_path):
            print(f'\t-> {dmap_path}')
            ConvertFile(rawacf_path, dmap_path, "rawacf", "rawacf", 'site', 'dmap')

