# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains functions for converting bfiq files
to rawacf files.
"""

import numpy as np

try:
    import cupy as xp
except ImportError:
    import numpy as xp
    cupy_available = False
else:
    cupy_available = True


def correlations_from_samples(beamformed_samples_1, beamformed_samples_2, output_sample_rate,
                              slice_index_details):
    """
    Correlate two sets of beamformed samples together. Correlation matrices are used and
    indices corresponding to lag pulse pairs are extracted.

    :param      beamformed_samples_1:  The first beamformed samples.
    :type       beamformed_samples_1:  ndarray [num_slices, num_beams, num_samples]
    :param      beamformed_samples_2:  The second beamformed samples.
    :type       beamformed_samples_2:  ndarray [num_slices, num_beams, num_samples]
    :param      slice_index_details:   Details used to extract indices for each slice.
    :type       slice_index_details:   list

    :returns:   Correlations for slices.
    :rtype:     list
    """

    # [num_slices, num_beams, num_samples]
    # [num_slices, num_beams, num_samples]
    correlated = xp.einsum('ijk,ijl->ijkl', beamformed_samples_1.conj(),
                           beamformed_samples_2)

    if cupy_available:
        correlated = xp.asnumpy(correlated)

    values = []
    for s in slice_index_details:
        if s['lags'].size == 0:
            values.append(np.array([]))
            continue
        range_off = np.arange(s['num_range_gates'], dtype=np.int32) + s['first_range_off']

        tau_in_samples = s['tau_spacing'] * 1e-6 * output_sample_rate

        lag_pulses_as_samples = np.array(s['lags'], np.int32) * np.int32(tau_in_samples)

        # [num_range_gates, 1, 1]
        # [1, num_lags, 2]
        samples_for_all_range_lags = (range_off[...,np.newaxis,np.newaxis] +
                                      lag_pulses_as_samples[np.newaxis,:,:])

        # [num_range_gates, num_lags, 2]
        row = samples_for_all_range_lags[...,1].astype(np.int32)

        # [num_range_gates, num_lags, 2]
        column = samples_for_all_range_lags[...,0].astype(np.int32)

        values_for_slice = correlated[s['slice_num'],:,row,column]

        # [num_range_gates, num_lags, num_beams]
        values_for_slice = np.einsum('ijk->kij', values_for_slice)

        # [num_beams, num_range_gates, num_lags]
        values.append(values_for_slice)

    return values


def autocorrelate_record(record):
    """
    Takes a record from a bfiq file and calculates the autocorrelations.

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