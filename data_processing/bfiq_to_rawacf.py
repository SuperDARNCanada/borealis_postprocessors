# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller
"""
This file contains functions for converting bfiq files
to rawacf files.
"""
import itertools
import logging
import numpy as np
import os
import subprocess as sp
import deepdish as dd
from scipy.constants import speed_of_light

try:
    import cupy as xp
except ImportError:
    import numpy as xp
    cupy_available = False
else:
    cupy_available = True

postprocessing_logger = logging.getLogger('borealis_postprocessing')


def correlations_from_samples(beamformed_samples_1, beamformed_samples_2, record):
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

    # beamformed_samples_1: [num_beams, num_samples]
    # beamformed_samples_2: [num_beams, num_samples]
    # correlated:           [num_beams, num_samples, num_samples]
    correlated = xp.einsum('jk,jl->jlk', beamformed_samples_1.conj(),
                           beamformed_samples_2)

    if cupy_available:
        correlated = xp.asnumpy(correlated)

    values = []
    if record['lags'].size == 0:
        values.append(np.array([]))
        return values

    # First range offset in samples
    sample_off = record['first_range_rtt'] * 1e-6 * record['rx_sample_rate']
    sample_off = np.int32(sample_off)

    # Helpful values converted to units of samples
    range_off = np.arange(record['num_ranges'], dtype=np.int32) + sample_off
    tau_in_samples = record['tau_spacing'] * 1e-6 * record['rx_sample_rate']
    lag_pulses_as_samples = np.array(record['lags'], np.int32) * np.int32(tau_in_samples)

    # [num_range_gates, 1, 1]
    # [1, num_lags, 2]
    samples_for_all_range_lags = (range_off[...,np.newaxis,np.newaxis] +
                                  lag_pulses_as_samples[np.newaxis,:,:])

    # [num_range_gates, num_lags, 2]
    row = samples_for_all_range_lags[...,1].astype(np.int32)

    # [num_range_gates, num_lags, 2]
    column = samples_for_all_range_lags[...,0].astype(np.int32)

    # [num_beams, num_range_gates, num_lags]
    values = correlated[:,row,column]

    # Find the sample that corresponds to the second pulse transmitting
    second_pulse_sample_num = np.int32(tau_in_samples) * record['pulses'][1] - sample_off - 1

    # Replace all ranges which are contaminated by the second pulse for lag 0
    # with the data from those ranges after the final pulse.
    values[:, second_pulse_sample_num:, 0] = values[:, second_pulse_sample_num:, -1]

    return values


def convert_record(record, averaging_method):
    """
    Takes a record from a bfiq file and processes it into record for rawacf file.

    :param record:              Borealis bfiq record
    :param averaging_method:    Either 'mean' or 'median'
    :return:                    Record of rawacf data for rawacf site file
    """
    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------- Averaging Method ------------------------------------------------ #
    # ---------------------------------------------------------------------------------------------------------------- #
    record['averaging_method'] = averaging_method

    # ---------------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------- Correlate the data ----------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    pulse_phase_offset = record['pulse_phase_offset']
    if pulse_phase_offset is None:
        pulse_phase_offset = [0.0] * len(record['pulses'])

    # bfiq data shape  = [num_arrays, num_sequences, num_beams, num_samps]
    bfiq_data = record['data']

    # Get the data and reshape
    num_arrays, num_sequences, num_beams, num_samps = record['data_dimensions']
    bfiq_data = bfiq_data.reshape(record['data_dimensions'])

    num_lags = len(record['lags'])
    main_corrs_unavg = np.zeros((num_sequences, num_beams, record['num_ranges'], num_lags), dtype=np.complex64)
    intf_corrs_unavg = np.zeros((num_sequences, num_beams, record['num_ranges'], num_lags), dtype=np.complex64)
    cross_corrs_unavg = np.zeros((num_sequences, num_beams, record['num_ranges'], num_lags), dtype=np.complex64)

    # Loop through every sequence and compute correlations.
    # Output shape after loop is [num_sequences, num_beams, num_range_gates, num_lags]
    for sequence in range(num_sequences):
        # data input shape  = [num_antenna_arrays, num_beams, num_samps]
        # data return shape = [num_beams, num_range_gates, num_lags]
        main_corrs_unavg[sequence, ...] = correlations_from_samples(bfiq_data[0, sequence, :, :],
                                                                    bfiq_data[0, sequence, :, :],
                                                                    record)
        intf_corrs_unavg[sequence, ...] = correlations_from_samples(bfiq_data[1, sequence, :, :],
                                                                    bfiq_data[1, sequence, :, :],
                                                                    record)
        cross_corrs_unavg[sequence, ...] = correlations_from_samples(bfiq_data[0, sequence, :, :],
                                                                     bfiq_data[1, sequence, :, :],
                                                                     record)

    if averaging_method == 'median':
        main_corrs = np.median(np.real(main_corrs_unavg), axis=0) + 1j * np.median(np.imag(main_corrs_unavg), axis=0)
        intf_corrs = np.median(np.real(intf_corrs_unavg), axis=0) + 1j * np.median(np.imag(intf_corrs_unavg), axis=0)
        cross_corrs = np.median(np.real(cross_corrs_unavg), axis=0) + 1j * np.median(np.imag(cross_corrs_unavg), axis=0)
    else:
        # Using mean averaging
        main_corrs = np.einsum('ijkl->jkl', main_corrs_unavg) / num_sequences
        intf_corrs = np.einsum('ijkl->jkl', intf_corrs_unavg) / num_sequences
        cross_corrs = np.einsum('ijkl->jkl', cross_corrs_unavg) / num_sequences

    record['main_acfs'] = main_corrs.flatten()
    record['intf_acfs'] = intf_corrs.flatten()
    record['xcfs'] = cross_corrs.flatten()

    # ---------------------------------------------------------------------------------------------------------------- #
    # --------------------------------------- Data Descriptors & Dimensions ------------------------------------------ #
    # ---------------------------------------------------------------------------------------------------------------- #
    record['correlation_descriptors'] = ['num_beams', 'num_ranges', 'num_lags']
    record['correlation_dimensions'] = np.array([num_beams, record['num_ranges'], num_lags],
                                                dtype=np.uint32)

    # ---------------------------------------------------------------------------------------------------------------- #
    # -------------------------------------------- Remove extra fields ----------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    del record['data']
    del record['data_descriptors']
    del record['data_dimensions']
    del record['num_ranges']
    del record['num_samps']
    del record['pulse_phase_offset']
    del record['antenna_arrays_order']

    # Fix representation of empty dictionary if no slice interfacing present
    slice_interfacing = record['slice_interfacing']
    if not isinstance(slice_interfacing, dict) and slice_interfacing == '{':
        record['slice_interfacing'] = '{}'

    return record


def bfiq_to_rawacf(infile, outfile, averaging_method):
    """
    Converts a bfiq site file to rawacf site file

    :param infile:              Borealis bfiq site file
    :type  infile:              String
    :param outfile:             Borealis bfiq site file
    :type  outfile:             String
    :param averaging_method:    Method to average over a sequence. Either 'mean' or 'median'
    :type  averaging_method:    String
    :return:                    Path to rawacf site file
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
        correlated_record = convert_record(group[record], averaging_method)

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
