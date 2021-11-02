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

    num_beams = len(beam_azms)
    wave_num = 2.0 * np.pi * freq / speed_of_light

    # Flatten num_antennas_1 with num_antennas_2 to produce vector snapshots in (range, lag)
    num_ant_1, num_ant_2, num_ranges, num_lags = correlations.shape
    elongated_corrs = np.resize(correlations, (num_ant_1 * num_ant_2, num_ranges, num_lags))

    # Create complex exponential matrix for inversion
    # -> [num_ant_1, num_ant_2]
    antenna_index_diffs = np.arange(num_ant_1, np.newaxis) - np.arange(np.newaxis, num_ant_2)

    # [num_beams]
    phases = np.sin(beam_azms)

    # [num_ant_1, num_ant_2, num_beams]
    exponents = np.einsum('ij,k->ijk', antenna_index_diffs, phases)
    phase_matrix = np.exp(-1j * wave_num * antenna_spacing * exponents)

    # [num_ant_1 * num_ant_2, num_beams]
    elongated_phase_matrix = np.resize(phase_matrix, (num_ant_1 * num_ant_2, num_beams))

    # TODO: Compare with scipy.linalg.pinv and scipy.linalg.lstsq
    # Solve the least squares problem with Moore-Penrose pseudo inverse method
    pseudo_inv = np.linalg.pinv(elongated_phase_matrix)

    # elongated_phase_matrix: [num_ant_1*num_ant_2, num_beams]
    # elongated_corrs:        [num_ant_1*num_ant_2, num_ranges, num_lags]
    # xsections:              [num_beams, num_ranges, num_lags]
    xsections = np.einsum('il,ijk,->ljk', elongated_phase_matrix, elongated_corrs)

    # # TODO: Compute covariance matrices
    # covars = np.ones((num_ant_1, num_ant_2, num_ranges, num_lags))
    # covars_inv = np.linalg.inv(covars.transpose((2, 3, 0, 1)))
    #
    # # TODO: Figure out this syntax (elongated_phase_matrix^H * covars_inv)
    # bh_cinv = np.einsum()
    # # TODO: Figure out this syntax (bh_cinv * elongated_phase_matrix)
    # mp_inv = np.einsum()
    # # TODO: Figure out this syntax (mp_inv^-1 * mp_inv)
    # xsections = np.einsum(np.linalg.inv(mp_inv), mp_inv)

    return xsections


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
    values_for_slice = correlated[:, :, row, column]

    return values


def image_record(record, num_bins, min_angle, max_angle):
    """
    Images a record into num_bins between min_angle and max_angle

    :param record:      Record from antennas_iq file. Dictionary
    :param num_bins:    Number of azimuthal bins. Int
    :param min_angle:   Minimum angle, clockwise from boresight. Degrees
    :param max_angle:   Maximum angle, clockwise from boresight. Degrees
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
    # TODO: Figure out how to get this
    averaging_method = 'mean'
    record['averaging_method'] = averaging_method

    # ---------------------------------------------------------------------------------------------------------------- #
    # ----------------------------------------------- Image the data ------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    beam_azms = record['beam_azms']
    freq = record['freq']
    pulse_phase_offset = record['pulse_phase_offset']
    if pulse_phase_offset is None:
        pulse_phase_offset = [0.0] * len(record['pulses'])

    # antennas data shape  = [num_antennas, num_sequences, num_samps]
    antennas_data = record['data']

    # Get the data and reshape
    num_antennas, num_sequences, num_samps = record['data_dimensions']
    antennas_data = antennas_data.reshape(record['data_dimensions'])

    # Loop through antennas and calculate correlation matrices, averaging over all sequences
    main_imaged_data = np.array([], dtype=np.complex64)
    intf_imaged_data = np.array([], dtype=np.complex64)
    main_antenna_count = record['main_antenna_count']

    # TODO: Grab these values from somewhere
    main_antenna_spacing = 15.24
    intf_antenna_spacing = 15.24

    tau_in_samples = record['tau_spacing'] * 1e-6 * record['rx_sample_rate']
    lag_pulses_as_samples = np.array(record['lags'], np.int32) * np.int32(tau_in_samples)

    pulse_phase_offset = record['pulse_phase_offset']
    if pulse_phase_offset is None:
        pulse_phase_offset = [0.0] * len(record['pulses'])

    num_lags = len(record['lags'])
    main_corrs_unavg = np.zeros((num_sequences, num_antennas, num_antennas,
                                 record['num_ranges'], num_lags), dtype=np.complex64)
    intf_corrs_unavg = np.zeros((num_sequences, num_antennas, num_antennas,
                                 record['num_ranges'], num_lags), dtype=np.complex64)
    cross_corrs_unavg = np.zeros((num_sequences, num_antennas, num_antennas,
                                  record['num_ranges'], num_lags), dtype=np.complex64)

    # Loop through every sequence and compute correlations.
    # Output shape after loop is [num_sequences, num_antennas, num_antennas, num_range_gates, num_lags]
    for sequence in range(num_sequences):
        # data input shape  = [num_antennas, num_samps]
        # data return shape = [num_antennas, num_antennas, num_range_gates, num_lags]
        main_corrs_unavg[sequence, ...] = correlations_from_samples(antennas_data[:main_antenna_count, sequence, :],
                                                                    antennas_data[:main_antenna_count, sequence, :],
                                                                    record)
        intf_corrs_unavg[sequence, ...] = correlations_from_samples(antennas_data[main_antenna_count:, sequence, :],
                                                                    antennas_data[main_antenna_count:, sequence, :],
                                                                    record)
        cross_corrs_unavg[sequence, ...] = correlations_from_samples(antennas_data[:main_antenna_count, sequence, :],
                                                                     antennas_data[main_antenna_count:, sequence, :],
                                                                     record)
    if averaging_method == 'median':
        # TODO: Sort first
        main_corrs = main_corrs_unavg[num_sequences // 2, ...]
        intf_corrs = intf_corrs_unavg[num_sequences // 2, ...]
        cross_corrs = cross_corrs_unavg[num_sequences // 2, ...]
    else:
        # Using mean averaging
        main_corrs = np.einsum('ijklm->jklm', main_corrs_unavg) / num_sequences
        intf_corrs = np.einsum('ijklm->jklm', intf_corrs_unavg) / num_sequences
        cross_corrs = np.einsum('ijklm->jklm', cross_corrs_unavg) / num_sequences

    # Use least squares inversion to estimate the scattering cross-section
    main_xsections = least_squares_inversion(main_corrs, beam_azms, freq, main_antenna_spacing)
    intf_xsections = least_squares_inversion(intf_corrs, beam_azms, freq, intf_antenna_spacing)

    record['main_cross_sections'] = main_xsections.flatten()
    record['intf_xsections'] = intf_xsections.flatten()

    # TODO: Figure out how to do cross-correlation (antenna spacings are different)
    #record['xcfs'] = cross_corrs.flatten()

    # ---------------------------------------------------------------------------------------------------------------- #
    # --------------------------------------- Data Descriptors & Dimensions ------------------------------------------ #
    # ---------------------------------------------------------------------------------------------------------------- #
    record['cross_section_descriptors'] = ['num_beams', 'num_ranges', 'num_lags']
    record['cross_section_dimensions'] = np.array([num_bins, record['num_ranges'], num_lags],
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

    # Fix representation of empty dictionary if no slice interfacing present
    slice_interfacing = record['slice_interfacing']
    if not isinstance(slice_interfacing, dict) and slice_interfacing == '{':
        record['slice_interfacing'] = '{}'

    return record


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
        imaged_record = image_record(group[record], num_bins, min_angle, max_angle)

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