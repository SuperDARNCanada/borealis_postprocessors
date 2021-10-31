# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains functions to image antennas_iq data into
rawacf-style data with higher azimuthal resolution than
standard SuperDARN rawacf files.
"""
import itertools
import math
import os
import subprocess as sp
import numpy as np
import deepdish as dd
from scipy.constants import speed_of_light

try:
    import cupy as xp
except ImportError:
    import numpy as xp

    cupy_available = False
else:
    cupy_available = True

import logging

postprocessing_logger = logging.getLogger('borealis_postprocessing')


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
    # ----------------------------------------------- Image the data ------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    beam_azms = record['beam_azms']
    freq = record['freq']
    beta = 2 * math.pi * freq / speed_of_light
    pulse_phase_offset = record['pulse_phase_offset']
    if pulse_phase_offset is None:
        pulse_phase_offset = [0.0] * len(record['pulses'])

    # antennas data shape  = [num_antennas, num_sequences, num_samps]
    antennas_data = record['data']

    # Get the data and reshape
    num_antennas, num_sequences, num_samps = record['data_dimensions']
    antennas_data = antennas_data.reshape(record['data_dimensions'])

    main_imaged_data = np.array([], dtype=np.complex64)
    intf_imaged_data = np.array([], dtype=np.complex64)
    main_antenna_count = record['main_antenna_count']

    # TODO: Grab these values from somewhere
    main_antenna_spacing = 15.24
    intf_antenna_spacing = 15.24

    # Loop through every sequence and image the data.
    # Output shape after loop is [num_sequences, num_beams, num_samps]
    for sequence in range(num_sequences):
        # data input shape  = [num_antennas, num_samps]
        # data return shape = [num_beams, num_samps]
        main_imaged_data = np.append(main_imaged_data,
                                     image(antennas_data[:main_antenna_count, sequence, :],
                                           beam_azms,
                                           freq,
                                           main_antenna_spacing,
                                           pulse_phase_offset))
        intf_imaged_data = np.append(intf_imaged_data,
                                     image(antennas_data[main_antenna_count:, sequence, :],
                                           beam_azms,
                                           freq,
                                           intf_antenna_spacing,
                                           pulse_phase_offset))

    record['data'] = np.append(main_imaged_data, intf_imaged_data).flatten()

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
