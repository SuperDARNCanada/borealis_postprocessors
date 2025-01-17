"""
This script exists to fix array-structured files with improper pulse_phase_offset fields.
"""

import glob
import os
import argparse
import h5py
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', help='Directory containing files to process')
    parser.add_argument('--pattern', help='Pattern to search for when globbing files from indir',
                        default='*antennas_iq.hdf5.site')
    args = parser.parse_args()
    in_directory = args.indir
    pattern = args.pattern

    for path in glob.glob(f'{in_directory}/{pattern}'):
        if "antennas_iq" not in path:
            continue
        if not os.path.isfile(path):
            continue
        if path.endswith(".site"):
            input_structure = "site"
        else:
            input_structure = "array"

        print(f'{path}')

        with h5py.File(path, 'r+') as f:
            ppo = f['pulse_phase_offset'][()]
            tstamps = f['sqn_timestamps'][()]
            if input_structure == 'array':
                num_recs = tstamps.shape[0]
                if ppo.size <= 2:
                    del f['pulse_phase_offset']
                    f.create_dataset('pulse_phase_offset', data=np.zeros((num_recs, 1), dtype=np.float32))
            else:
                print("Not sure how to handle site yet")
