# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains functions for converting bfiq files
to rawacf files.
"""

import logging
import numpy as np
import os
import subprocess as sp
import deepdish as dd

try:
    import cupy as xp
except ImportError:
    import numpy as xp
    cupy_available = False
else:
    cupy_available = True

postprocessing_logger = logging.getLogger('borealis_postprocessing')


def correlations_from_samples(beamformed_samples_1, beamformed_samples_2, output_sample_rate,
                              record):
    """
    Correlate two sets of beamformed samples together. Correlation matrices are used and
    indices corresponding to lag pulse pairs are extracted.

    :param      beamformed_samples_1:  The first beamformed samples.
    :type       beamformed_samples_1:  ndarray [num_slices, num_beams, num_samples]
    :param      beamformed_samples_2:  The second beamformed samples.
    :type       beamformed_samples_2:  ndarray [num_slices, num_beams, num_samples]
    :param      record:                Details used to extract indices for each slice.
    :type       record:                dictionary

    :returns:   Correlations.
    :rtype:     list
    """

    # [num_beams, num_samples]
    # [num_beams, num_samples]
    correlated = xp.einsum('ij,jl->jkl', beamformed_samples_1.conj(),
                           beamformed_samples_2)

    if cupy_available:
        correlated = xp.asnumpy(correlated)

    values = []
    if record['lags'].size == 0:
        values.append(np.array([]))
        return values

    # First range offset in samples
    sample_off = record['first_range_rtt'] * 1e-6 * output_sample_rate
    sample_off = np.int32(sample_off)

    # Range offset in samples
    range_off = np.arange(record['num_ranges'], dtype=np.int32) + sample_off

    tau_in_samples = record['tau_spacing'] * 1e-6 * output_sample_rate

    lag_pulses_as_samples = np.array(record['lags'], np.int32) * np.int32(tau_in_samples)

    # [num_range_gates, 1, 1]
    # [1, num_lags, 2]
    samples_for_all_range_lags = (range_off[...,np.newaxis,np.newaxis] +
                                  lag_pulses_as_samples[np.newaxis,:,:])

    # [num_range_gates, num_lags, 2]
    row = samples_for_all_range_lags[...,1].astype(np.int32)

    # [num_range_gates, num_lags, 2]
    column = samples_for_all_range_lags[...,0].astype(np.int32)

    values_for_record = correlated[:,row,column]

    # [num_range_gates, num_lags, num_beams]
    values = np.einsum('ijk->kij', values_for_record)

    return values


def convert_record(record):
    """
    Takes a record from a bfiq file and processes it into record for rawacf file.

    :param record:      Borealis bfiq record
    :return:            Record of rawacf data for rawacf site file
    """


def bfiq_to_rawacf(infile, outfile):
    """
    Converts a bfiq site file to rawacf site file

    :param infile:      Borealis bfiq site file
    :type  infile:      String
    :param outfile:     Borealis bfiq site file
    :type  outfile:     String
    :return:            Path to rawacf site file
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
        correlated_record = convert_record(group[record])

        # Convert to numpy arrays for saving to file with deepdish
        formatted_record = convert_to_numpy(correlated_record)

        # Save record to temporary file
        tempfile = '/tmp/{}.tmp'.format(record)
        dd.io.save(tempfile, formatted_record, compression=None)

        # Copy record to output file
        cmd = 'h5copy -i {} -o {} -s {} -d {}'
        cmd = cmd.format(tempfile, outfile, '/', '/{}'.format(record))
        sp.call(cmd.split())

        # Remove temporary file
        os.remove(tempfile)
