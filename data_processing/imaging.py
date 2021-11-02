# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains functions to image antennas_iq data into
rawacf-style data with higher azimuthal resolution than
standard SuperDARN rawacf files.

Performs imaging algorithm from Bristow 2019 (https://doi.org/10.1029/2019RS006851)
"""
import itertools
import math
import os
import subprocess as sp
import numpy as np
import deepdish as dd
from scipy.constants import speed_of_light
from scipy.signal import correlate

try:
    import cupy as xp
except ImportError:
    import numpy as xp

    cupy_available = False
else:
    cupy_available = True

import logging

postprocessing_logger = logging.getLogger('borealis_postprocessing')


def least_squares_inversion(correlations, beam_azms, freq, antenna_spacing):
    """
    Calculates the least squares inversion of Ax = b, retrieving the correlation
    values at each beam, range, and lag.

    :param correlations:    Antenna correlation values for each range and lag
    :param beam_azms:       Beam directions. ndarray of floats (degrees)
    :param freq:            Transmit frequency. kHz
    :param antenna_spacing: Spacing between adjacent antennas. Meters.
    :return: The correlations for each beam, range, and lag
    """
    # Useful quantities
    beam_rads = np.radians(beam_azms)
    num_beams = len(beam_azms)
    wave_num = 2.0 * np.pi * (freq * 1000.0) / speed_of_light

    # Flatten num_antennas_1 with num_antennas_2 to produce vector snapshots in (range, lag)
    num_ant_1, num_ant_2, num_ranges, num_lags = correlations.shape
    elongated_corrs = np.resize(correlations, (num_ant_1 * num_ant_2, num_ranges, num_lags))

    # Create complex exponential matrix for inversion
    # -> [num_ant_1, num_ant_2]
    antenna_1_indices = np.arange(num_ant_1)
    antenna_2_indices = np.arange(num_ant_2)
    antenna_index_diffs = antenna_1_indices[:, np.newaxis] - antenna_2_indices[np.newaxis, :]

    # [num_beams]
    phases = np.sin(beam_rads)

    # [num_ant_1, num_ant_2, num_beams]
    exponents = np.einsum('ij,k->ijk', antenna_index_diffs, phases)
    phase_matrix = np.exp(-1j * wave_num * antenna_spacing * exponents)

    # [num_ant_1 * num_ant_2, num_beams]
    elongated_phase_matrix = np.resize(phase_matrix, (num_ant_1 * num_ant_2, num_beams))

    # TODO: Compare with scipy.linalg.pinv and scipy.linalg.lstsq
    # Solve the least squares problem with Moore-Penrose pseudo inverse method
    pseudo_inv = np.linalg.pinv(elongated_phase_matrix)

    # pseudo_inv:             [num_beams, num_ant_1*num_ant_2]
    # elongated_corrs:        [num_ant_1*num_ant_2, num_ranges, num_lags]
    # corrs:                  [num_beams, num_ranges, num_lags]
    corrs = np.einsum('li,ijk->ljk', pseudo_inv, elongated_corrs)

    return corrs


def correlations_from_samples(samples_1, samples_2, record):
    """
    Correlate two sets of samples together. Correlation matrices are used and
    indices corresponding to lag pulse pairs are extracted.

    :param      samples_1:  The first set of samples.
    :type       samples_1:  ndarray [num_antennas_1, num_samples]
    :param      samples_2:  The second set of samples.
    :type       samples_2:  ndarray [num_antennas_2, num_samples]
    :param      record:     Details used to extract indices for each slice.
    :type       record:     dictionary

    :returns:   Correlations for slices.
    :rtype:     list
    """

    # samples_1:    [num_antennas_1, num_samples]
    # samples_2:    [num_antennas_2, num_samples]
    # correlated:   [num_antennas_1, num_antennas_2, num_samples, num_samples]
    correlated = xp.einsum('ik,jl->ijkl', samples_1.conj(), samples_2)

    if cupy_available:
        correlated = xp.asnumpy(correlated)

    values = []

    # If there are no lag pairs, there is no useful data
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
    # -> [num_range_gates, num_lags, 2]
    samples_for_all_range_lags = (range_off[..., np.newaxis, np.newaxis] +
                                  lag_pulses_as_samples[np.newaxis, :, :])

    # -> [num_range_gates, num_lags, 1]
    row = samples_for_all_range_lags[..., 1].astype(np.int32)

    # -> [num_range_gates, num_lags, 1]
    column = samples_for_all_range_lags[..., 0].astype(np.int32)

    # -> [num_antennas_1, num_antennas_2, num_range_gates, num_lags
    values = correlated[:, :, row, column]

    return values


def image_record(record, num_bins, min_angle, max_angle, averaging_method):
    """
    Images a record into num_bins between min_angle and max_angle

    :param record:              Record from antennas_iq file. Dictionary
    :param num_bins:            Number of azimuthal bins. Int
    :param min_angle:           Minimum angle, clockwise from boresight. Degrees
    :param max_angle:           Maximum angle, clockwise from boresight. Degrees
    :param averaging_method:    Averaging method to use. 'mean' or 'median'
    :return:            Record with imaged data. Dictionary
    """
    # ---------------------------------------------------------------------------------------------------------------- #
    # ------------------------------------------------ First Range --------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    first_range = 180.0  # scf.FIRST_RANGE
    record['first_range'] = np.float32(first_range)

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------- First Range Round Trip Time ------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    first_range_rtt = first_range * 2.0 * 1.0e3 * 1e6 / speed_of_light
    record['first_range_rtt'] = np.float32(first_range_rtt)

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------- Create Lag Table ------------------------------------------------ #
    # ---------------------------------------------------------------------------------------------------------------- #
    lag_table = list(itertools.combinations(record['pulses'], 2))  # Create all combinations of lags
    lag_table.append([record['pulses'][0], record['pulses'][0]])  # lag 0
    lag_table = sorted(lag_table, key=lambda x: x[1] - x[0])  # sort by lag number
    lag_table.append([record['pulses'][-1], record['pulses'][-1]])  # alternate lag 0
    lags = np.array(lag_table, dtype=np.uint32)
    record['lags'] = lags

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------- Number of Ranges ------------------------------------------------ #
    # ---------------------------------------------------------------------------------------------------------------- #
    # TODO: Do this intelligently. Maybe grab from githash and cpid? Have default values too

    station = record['station']
    if station in ["cly", "rkn", "inv"]:
        num_ranges = 100  # scf.POLARDARN_NUM_RANGES
        record['num_ranges'] = np.uint32(num_ranges)
    elif station in ["sas", "pgr"]:
        num_ranges = 75  # scf.STD_NUM_RANGES
        record['num_ranges'] = np.uint32(num_ranges)

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------- Range Separation ------------------------------------------------ #
    # ---------------------------------------------------------------------------------------------------------------- #
    range_sep = 1 / record['rx_sample_rate'] * speed_of_light / 1.0e3 / 2.0
    record['range_sep'] = np.float32(range_sep)

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------- Averaging Method ------------------------------------------------ #
    # ---------------------------------------------------------------------------------------------------------------- #
    record['averaging_method'] = averaging_method

    # ---------------------------------------------------------------------------------------------------------------- #
    # ----------------------------------------------- Image the data ------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    tx_beam_azms = record['beam_azms']
    freq = record['freq']
    pulse_phase_offset = record['pulse_phase_offset']
    if pulse_phase_offset is None:
        pulse_phase_offset = [0.0] * len(record['pulses'])

    # antennas data shape  = [num_antennas, num_sequences, num_samps]
    antennas_data = record['data']

    # Get the data and reshape
    num_antennas, num_sequences, num_samps = record['data_dimensions']
    antennas_data = antennas_data.reshape(record['data_dimensions'])

    main_antenna_count = record['main_antenna_count']
    intf_antenna_count = record['intf_antenna_count']

    # TODO: Grab these values from somewhere
    main_antenna_spacing = 15.24
    intf_antenna_spacing = 15.24

    num_lags = len(record['lags'])

    main_antenna_corrs_unavg = np.zeros((num_sequences, main_antenna_count, main_antenna_count,
                                         record['num_ranges'], num_lags), dtype=np.complex64)
    intf_antenna_corrs_unavg = np.zeros((num_sequences, intf_antenna_count, intf_antenna_count,
                                         record['num_ranges'], num_lags), dtype=np.complex64)
    cross_antenna_corrs_unavg = np.zeros((num_sequences, main_antenna_count, intf_antenna_count,
                                          record['num_ranges'], num_lags), dtype=np.complex64)

    # Loop through every sequence and compute antenna correlations.
    # Output shape after loop is [num_sequences, num_antennas, num_antennas, num_range_gates, num_lags]
    for sequence in range(num_sequences):
        # data input shape  = [num_antennas, num_samps]
        # data return shape = [num_antennas, num_antennas, num_range_gates, num_lags]
        main_antenna_corrs_unavg[sequence, ...] = correlations_from_samples(
            antennas_data[:main_antenna_count, sequence, :],
            antennas_data[:main_antenna_count, sequence, :],
            record)
        intf_antenna_corrs_unavg[sequence, ...] = correlations_from_samples(
            antennas_data[main_antenna_count:, sequence, :],
            antennas_data[main_antenna_count:, sequence, :],
            record)
        cross_antenna_corrs_unavg[sequence, ...] = correlations_from_samples(
            antennas_data[:main_antenna_count, sequence, :],
            antennas_data[main_antenna_count:, sequence, :],
            record)

    # Average the correlations
    if averaging_method == 'median':
        main_antenna_corrs = np.median(main_antenna_corrs_unavg, axis=0)
        intf_antenna_corrs = np.median(intf_antenna_corrs_unavg, axis=0)
        cross_antenna_corrs = np.median(cross_antenna_corrs_unavg, axis=0)
    else:
        # Using mean averaging
        main_antenna_corrs = np.einsum('ijklm->jklm', main_antenna_corrs_unavg) / num_sequences
        intf_antenna_corrs = np.einsum('ijklm->jklm', intf_antenna_corrs_unavg) / num_sequences
        cross_antenna_corrs = np.einsum('ijklm->jklm', cross_antenna_corrs_unavg) / num_sequences

    # Create azimuth bins for imaging
    rx_beam_azms = np.arange(min_angle, max_angle + 1, (max_angle+1 - min_angle) / num_bins)
    record['rx_beam_azms'] = rx_beam_azms

    # Use least squares inversion to estimate the scattering cross-section
    main_corrs = least_squares_inversion(main_antenna_corrs, rx_beam_azms, freq, main_antenna_spacing)
    intf_corrs = least_squares_inversion(intf_antenna_corrs, rx_beam_azms, freq, intf_antenna_spacing)

    record['main_acfs'] = main_corrs.flatten()
    record['intf_acfs'] = intf_corrs.flatten()

    # TODO: Figure out how to do cross-correlation (antenna spacings are different)
    #record['xcfs'] = cross_corrs.flatten()

    # ---------------------------------------------------------------------------------------------------------------- #
    # --------------------------------------- Data Descriptors & Dimensions ------------------------------------------ #
    # ---------------------------------------------------------------------------------------------------------------- #
    record['correlation_descriptors'] = ['num_beams', 'num_ranges', 'num_lags']
    record['correlation_dimensions'] = np.array([num_bins, record['num_ranges'], num_lags],
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

    # Rename beam_azms to reflect that it is the transmit beam azimuths
    record['tx_beam_azms'] = tx_beam_azms
    record['tx_beam_nums'] = record['beam_nums']
    del record['beam_azms']
    del record['beam_nums']

    # Fix representation of empty dictionary if no slice interfacing present
    slice_interfacing = record['slice_interfacing']
    if not isinstance(slice_interfacing, dict) and slice_interfacing == '{':
        record['slice_interfacing'] = '{}'

    return record


def image_data(infile, outfile, num_bins, min_angle, max_angle, averaging_method):
    """
    Performs imaging on data from infile, for num_bins azimuthal angles
    between min_angle and max_angle.

    :param infile:              Name of antennas_iq file to image. String
    :param outfile:             Name of file where imaged data will be stored. String
    :param num_bins:            Number of azimuthal bins. Int
    :param min_angle:           Minimum angle to right of boresight. Degrees
    :param max_angle:           Maximum angle to right of boresight. Degrees
    :param averaging_method:    Averaging method to use. 'mean' or 'median'
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

    # TODO: Verify inputs are reasonable
    # Load file to read in records
    group = dd.io.load(infile)
    records = group.keys()

    # Convert each record to bfiq record
    for record in records:
        imaged_record = image_record(group[record], num_bins, min_angle, max_angle, averaging_method)

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
