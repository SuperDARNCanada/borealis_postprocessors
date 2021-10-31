# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains functions to image antennas_iq data into
rawacf-style data with higher azimuthal resolution than
standard SuperDARN rawacf files.
"""

import os
import subprocess as sp
import numpy as np
import deepdish as dd

try:
    import cupy as xp
except ImportError:
    import numpy as xp

    cupy_available = False
else:
    cupy_available = True

import logging

postprocessing_logger = logging.getLogger('borealis_postprocessing')


def image_data(infile, outfile, num_bins, min_angle, max_angle):
    """
    Performs imaging on data from infile, for num_bins azimuthal angles
    between min_angle and max_angle.

    :param infile:      Name of antennas_iq file to image. String
    :param outfile:     Name of file where imaged data will be stored. String
    :param num_bins:    Number of azimuthal bins. Int
    :param min_angle:   Minimum angle to right of boresight. Degrees
    :param max_angle:   Maximum angle to right of boresight. Degrees
    :return:
    """
    def convert_to_numpy(data):
        """Converts lists stored in dict into numpy array. Recursive.
        Args:
            data (Python dictionary): Dictionary with lists to convert to numpy arrays.
        """
        for k, v in data.items():
            if isinstance(v, dict):
                convert_to_numpy(v)
            elif isinstance(v, list):
                data[k] = np.array(v)
            else:
                continue
        return data

    postprocessing_logger.info('Converting file {} to bfiq'.format(infile))

    # Load file to read in records
    group = dd.io.load(infile)
    records = group.keys()

    # Convert each record to bfiq record
    for record in records:
        imaged_record = image_record(group[record])

        # Convert to numpy arrays for saving to file with deepdish
        formatted_record = convert_to_numpy(imaged_record)

        # Save record to temporary file
        tempfile = '/tmp/{}.tmp'.format(record)
        dd.io.save(tempfile, formatted_record, compression=None)

        # Copy record to output file
        cmd = 'h5copy -i {} -o {} -s {} -d {}'
        cmd = cmd.format(tempfile, outfile, '/', '/{}'.format(record))
        sp.call(cmd.split())

        # Remove temporary file
        os.remove(tempfile)
