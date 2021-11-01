# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
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

    # beamformed_samples_1: [num_beams, num_samples]
    # beamformed_samples_2: [num_beams, num_samples]
    # correlated:           [num_beams, num_samples, num_samples]
    correlated = xp.einsum('ij,ik->ijk', beamformed_samples_1.conj(),
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

    # Helpful values converted to units of samples
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

    # [num_range_gates, num_lags, num_beams]
    values_for_record = correlated[:,row,column]

    # [num_beams, num_range_gates, num_lags]
    values = np.einsum('ijk->kij', values_for_record)

    return values


def convert_record(record):
    """
    Takes a record from a bfiq file and processes it into record for rawacf file.

    :param record:      Borealis bfiq record
    :return:            Record of rawacf data for rawacf site file
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
    # ---------------------------------------------- Beamform the data ----------------------------------------------- #
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

    main_beamformed_data = np.array([], dtype=np.complex64)
    intf_beamformed_data = np.array([], dtype=np.complex64)
    main_antenna_count = record['main_antenna_count']

    # TODO: Grab these values from somewhere
    main_antenna_spacing = 15.24
    intf_antenna_spacing = 15.24

    # Loop through every sequence and beamform the data.
    # Output shape after loop is [num_sequences, num_beams, num_samps]
    for sequence in range(num_sequences):
        # data input shape  = [num_antennas, num_samps]
        # data return shape = [num_beams, num_samps]
        main_beamformed_data = np.append(main_beamformed_data,
                                         beamform(antennas_data[:main_antenna_count, sequence, :],
                                                  beam_azms,
                                                  freq,
                                                  main_antenna_spacing,
                                                  pulse_phase_offset))
        intf_beamformed_data = np.append(intf_beamformed_data,
                                         beamform(antennas_data[main_antenna_count:, sequence, :],
                                                  beam_azms,
                                                  freq,
                                                  intf_antenna_spacing,
                                                  pulse_phase_offset))

    record['data'] = np.append(main_beamformed_data, intf_beamformed_data).flatten()

    # ---------------------------------------------------------------------------------------------------------------- #
    # --------------------------------------- Data Descriptors & Dimensions ------------------------------------------ #
    # ---------------------------------------------------------------------------------------------------------------- #
    # Old dimensions: [num_antennas, num_sequences, num_samps]
    # New dimensions: [num_antenna_arrays, num_sequences, num_beams, num_samps]
    data_descriptors = record['data_descriptors']
    record['data_descriptors'] = ['num_antenna_arrays',
                                  data_descriptors[1],
                                  'num_beams',
                                  data_descriptors[2]]
    record['data_dimensions'] = np.array([2, num_sequences, len(beam_azms), num_samps],
                                         dtype=np.uint32)

    # ---------------------------------------------------------------------------------------------------------------- #
    # -------------------------------------------- Antennas Array Order ---------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    record['antenna_arrays_order'] = ['main', 'intf']

    return record

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
