# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains functions to image antennas_iq data into rawacf-style data without using traditional beamforming.

Performs imaging algorithm from Bristow 2019 (https://doi.org/10.1029/2019RS006851)
"""
from collections import OrderedDict
import itertools
import logging
from typing import Union

import numpy as np
from scipy.constants import speed_of_light

from postprocessors import BaseConvert
try:
    import cupy as xp
except ImportError:
    import numpy as xp

    cupy_available = False
else:
    cupy_available = True

postprocessing_logger = logging.getLogger('borealis_postprocessing')


class BristowImaging(BaseConvert):
    """
    Class for conversion of Borealis antennas_iq files into rawacf files. This class inherits from
    BaseConvert, which handles all functionality generic to postprocessing borealis files.

    See Also
    --------
    ConvertFile
    BaseConvert
    ProcessAntennasIQ2Rawacf

    Attributes
    ----------
    infile: str
        The filename of the input antennas_iq file.
    outfile: str
        The file name of output file
    infile_structure: str
        The structure of the file. Structures include:
        'array'
        'site'
    outfile_structure: str
        The desired structure of the output file. Same structures as
        above, with the addition of 'dmap'.
    averaging_method: str
        Averaging method for computing correlations (for processing into rawacf files).
        Acceptable values are 'mean' and 'median'.
    """

    def __init__(self, infile: str, outfile: str, infile_structure: str, outfile_structure: str,
                 num_bins: int = 60, min_angle: float = -30., max_angle: float = 30., **kwargs):
        """
        Initialize the attributes of the class.

        Parameters
        ----------
        infile: str
            Path to input file.
        outfile: str
            Path to output file.
        infile_structure: str
            Borealis structure of input file. Either 'array' or 'site'.
        outfile_structure: str
            Borealis structure of output file. Either 'array', 'site', or 'dmap'.
        num_bins: int
            Number of angle bins for imaging. Default 60.
        min_angle: float
            Minimum angle CW of boresight for imaging. Default -30
        max_angle: float
            Maximum angle CW of boresight for imaging. Default 30
        **kwargs: dict
            Any additional keyword arguments for processing.
            Kwargs specific to this class are:
                averaging_method: str
                    Method for averaging correlations across sequences. Either 'median' or 'mean'. Default 'mean'.
        """
        super().__init__(infile, outfile, 'bfiq', 'rawacf', infile_structure, outfile_structure)
        self.averaging_method = kwargs.get('averaging_method', 'mean')  # Default to 'mean'
        self.process_file(num_bins=num_bins, min_angle=min_angle, max_angle=max_angle, **kwargs)

    @staticmethod
    def process_record(record: OrderedDict, averaging_method: Union[None, str], **kwargs) -> OrderedDict:
        """
        Takes a record from a bfiq file and processes it into record for rawacf file.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing bfiq data and metadata
        averaging_method: Union[None, str]
            Averaging method to use. Supported methods are 'mean' and 'median'.

        Returns
        -------
        record: OrderedDict
            record converted to rawacf format
        """
        num_bins = kwargs['num_bins']
        min_angle = kwargs['min_angle']
        max_angle = kwargs['max_angle']

        first_range = 180.0  # scf.FIRST_RANGE
        record['first_range'] = np.float32(first_range)

        first_range_rtt = first_range * 2.0 * 1.0e3 * 1e6 / speed_of_light
        record['first_range_rtt'] = np.float32(first_range_rtt)

        lag_table = list(itertools.combinations(record['pulses'], 2))  # Create all combinations of lags
        lag_table.append([record['pulses'][0], record['pulses'][0]])  # lag 0
        lag_table = sorted(lag_table, key=lambda x: x[1] - x[0])  # sort by lag number
        lag_table.append([record['pulses'][-1], record['pulses'][-1]])  # alternate lag 0
        lags = np.array(lag_table, dtype=np.uint32)
        record['lags'] = lags

        # TODO: Do this intelligently. Maybe grab from githash and cpid? Have default values too
        record['num_ranges'] = np.uint32(75)

        range_sep = 1 / record['rx_sample_rate'] * speed_of_light / 1.0e3 / 2.0
        record['range_sep'] = np.float32(range_sep)

        record['averaging_method'] = averaging_method

        tx_beam_azms = record['beam_azms']
        freq = record['freq']

        # antennas data shape  = [num_antennas, num_sequences, num_samps]
        antennas_data = record['data']

        # Get the data and reshape
        num_antennas, num_sequences, num_samps = record['data_dimensions']
        antennas_data = antennas_data.reshape(record['data_dimensions'])

        main_antenna_count = record['main_antenna_count']
        # intf_antenna_count = record['intf_antenna_count']

        # TODO: Grab these values from somewhere
        main_antenna_spacing = 15.24
        intf_antenna_spacing = 15.24

        num_lags = len(record['lags'])

        # Create azimuth bins for imaging
        rx_beam_azms = np.arange(min_angle, max_angle + 1, (max_angle + 1 - min_angle) / num_bins)
        record['beam_azms'] = rx_beam_azms
        record['beam_nums'] = np.arange(len(rx_beam_azms), dtype=np.uint32)

        main_antenna_corrs_unavg = np.zeros((num_sequences, main_antenna_count*main_antenna_count, record['num_ranges'],
                                             num_lags), dtype=np.complex64)
        # intf_antenna_corrs_unavg = np.zeros((num_sequences, intf_antenna_count, record['num_ranges'], num_lags),
        #                                     dtype=np.complex64)

        # Compute antenna correlations.
        # Output shape afterwards is [num_sequences, num_antennas*num_antennas, num_range_gates, num_lags]
        main_antenna_corrs_unavg = BristowImaging.correlations_from_samples(antennas_data[:main_antenna_count],
                                                                            record)
        # intf_antenna_corrs_unavg = BristowImaging.correlations_from_samples(antennas_data[main_antenna_count:],
        #                                                                     record)

        # Apply taper to the correlations
        window = np.array([0.08081232549588463, 0.12098514265395757, 0.23455777475180511, 0.4018918165398586,
                           0.594054435182454, 0.7778186328978896, 0.9214100134552521, 1.0,
                           1.0, 0.9214100134552521, 0.7778186328978896, 0.594054435182454,
                           0.4018918165398586, 0.23455777475180511, 0.12098514265395757, 0.08081232549588463])
        square_window = np.minimum.outer(window, window).reshape((-1,))
        main_antenna_corrs_unavg = np.einsum('sarl,a->sarl', main_antenna_corrs_unavg, square_window)

        # Use least squares inversion to estimate the scattering cross-section
        # [num_beams, num_ranges, num_lags]
        main_corrs = BristowImaging.least_squares_inversion(main_antenna_corrs_unavg, rx_beam_azms, freq,
                                                            main_antenna_spacing)
        # intf_corrs = BristowImaging.least_squares_inversion(intf_antenna_corrs_unavg, rx_beam_azms, freq,
        #                                                     intf_antenna_spacing)

        record['main_acfs'] = main_corrs
        record['intf_acfs'] = np.zeros(main_corrs.shape, dtype=main_corrs.dtype)

        # TODO: Figure out how to do cross-correlation (antenna spacings are different)
        record['xcfs'] = np.zeros(main_corrs.shape, dtype=main_corrs.dtype)

        # Set data descriptors and dimensions
        githash = record['borealis_git_hash'].split('-')[0].strip('v').split('.')  # v0.6.1-xxxxxx -> ['0', '6', '1']
        if githash[0] == '0' and int(githash[1]) < 7:
            record['correlation_descriptors'] = ['num_beams', 'num_ranges', 'num_lags']
            record['correlation_dimensions'] = np.array([num_bins, record['num_ranges'], num_lags],
                                                        dtype=np.uint32)
            del record['data_descriptors']
            del record['data_dimensions']
        else:
            record['data_descriptors'] = np.bytes_(['num_beams', 'num_ranges', 'num_lags'])
            record['data_dimensions'] = np.array([num_bins, record['num_ranges'], num_lags], dtype=np.uint32)

        # Remove extra fields
        del record['data']
        del record['num_ranges']
        del record['num_samps']
        del record['pulse_phase_offset']
        del record['antenna_arrays_order']

        # Rename beam_azms to reflect that it is the transmitted beam azimuths
        # record['tx_beam_azms'] = tx_beam_azms
        # record['tx_beam_nums'] = record['beam_nums']
        # del record['beam_azms']
        # del record['beam_nums']

        # Fix representation of empty dictionary if no slice interfacing present
        slice_interfacing = record['slice_interfacing']
        if not isinstance(slice_interfacing, dict) and slice_interfacing == '{':
            record['slice_interfacing'] = '{}'

        return record

    @staticmethod
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
        dtype = xp.complex64
        # Useful quantities
        beam_rads = np.radians(beam_azms)
        num_beams = len(beam_azms)
        wave_num = 2.0 * np.pi * (freq * 1000.0) / speed_of_light

        # Some useful dimensions
        num_sequences, num_ant_squared, num_ranges, num_lags = correlations.shape
        num_ant = int(round(np.sqrt(num_ant_squared)))

        correlations = xp.asarray(correlations, dtype=dtype)

        # [num_ranges, num_lags, num_ant*num_ant]
        avg_correlations = xp.einsum('sarl->rla', correlations, dtype=dtype) / num_sequences

        # # Get the mean correlation vector
        #
        # sum_corr = xp.einsum('sarl->rla', correlations, dtype=dtype)
        # sum_corr_conj = xp.einsum('sbrl->rlb', correlations.conj(), dtype=dtype)
        #
        # # Calculate the covariance matrix, E[XX^H] - E[X]E[X]^H
        # # [num_ranges, num_lags, num_ant*num_ant, num_ant*num_ant]
        # cov = (xp.einsum('sarl,sbrl->rlab', correlations, correlations.conj()) -
        #        xp.einsum('rla,rlb->rlab', sum_corr, sum_corr_conj) / num_sequences) / num_sequences
        #
        # cov_inv = xp.linalg.inv(cov)

        # Create complex exponential matrix for inversion
        # [num_ant, num_ant]
        antenna_indices = np.arange(num_ant, dtype=np.int32)
        antenna_index_diffs = np.float32(antenna_indices[:, np.newaxis] - antenna_indices[np.newaxis, :])

        # [num_beams]
        phases = np.sin(beam_rads, dtype=np.float32)

        # [num_ant, num_ant, num_beams]
        exponents = np.einsum('ab,c->abc', antenna_index_diffs, phases, dtype=np.float32)
        phase_matrix = np.exp(-1j * wave_num * antenna_spacing * exponents, dtype=np.complex64)

        # [num_ant * num_ant, num_beams]
        elongated_phase_matrix = np.reshape(phase_matrix, (num_ant_squared, num_beams))
        elongated_phase_matrix = xp.asarray(elongated_phase_matrix, dtype=dtype)


        # TODO: Compare with scipy.linalg.pinv and scipy.linalg.lstsq
        # Solve the least squares problem with Moore-Penrose pseudo inverse method
        pseudo_inv = np.linalg.pinv(elongated_phase_matrix)
        #
        # pseudo_inv:             [num_beams, num_ant*num_ant]
        # avg_correlations:       [num_ranges, num_lags, num_ant*num_ant]
        # corrs:                  [num_beams, num_ranges, num_lags]
        corrs = xp.einsum('ba,rla->brl', pseudo_inv, avg_correlations)

        # [num_ranges, num_lags, num_beams, num_ant*num_ant]
        # bh_cov_inv = xp.einsum('ca,rlab->rlcb', elongated_phase_matrix.T.conj(), cov_inv, dtype=dtype)
        #
        # # [num_ranges, num_lags, num_beams, num_beams]
        # first_term = xp.linalg.inv(xp.einsum('rlba,ac->rlbc', bh_cov_inv, elongated_phase_matrix))
        #
        # # [num_ranges, num_lags, num_beams]
        # second_term = xp.einsum('rlba,rla->rlb', bh_cov_inv, sum_corr/num_sequences, dtype=dtype)

        # # [beams, num_ranges, num_lags]
        # corrs = xp.einsum('rlbc,rlb->crl', first_term, second_term, dtype=dtype)

        if cupy_available:
            corrs = xp.asnumpy(corrs)

        return corrs

    @staticmethod
    def correlations_from_samples(samples, record):
        """
        Correlates a set of samples with itself. Correlation matrices are used and
        indices corresponding to lag pulse pairs are extracted.

        :param      samples:    The set of samples.
        :type       samples:    ndarray [num_antennas, num_sequences, num_samples]
        :param      record:     Details used to extract indices for each slice.
        :type       record:     dictionary

        :returns:   Correlations for slices.
        :rtype:     np.ndarray
        """

        # # samples_1:    [num_antennas_1, num_sequences, num_samples]
        # # samples_2:    [num_antennas_2, num_sequences, num_samples]
        # # correlated:   [num_antennas_1, num_antennas_2, num_samples, num_samples]
        # correlated = xp.einsum('isk,jsl->sijkl', samples_1.conj(), samples_2)
        #
        # if cupy_available:
        #     correlated = xp.asnumpy(correlated)
        #
        # values = []
        #
        # # If there are no lag pairs, there is no useful data
        # if record['lags'].size == 0:
        #     values.append(np.array([]))
        #     return values
        #
        # # First range offset in samples
        # sample_off = record['first_range_rtt'] * 1e-6 * record['rx_sample_rate']
        # sample_off = np.int32(sample_off)
        #
        # # Helpful values converted to units of samples
        # range_off = np.arange(record['num_ranges'], dtype=np.int32) + sample_off
        # tau_in_samples = record['tau_spacing'] * 1e-6 * record['rx_sample_rate']
        # lag_pulses_as_samples = np.array(record['lags'], np.int32) * np.int32(tau_in_samples)
        #
        # # [num_range_gates, 1, 1]
        # # [1, num_lags, 2]
        # # -> [num_range_gates, num_lags, 2]
        # samples_for_all_range_lags = (range_off[..., np.newaxis, np.newaxis] +
        #                               lag_pulses_as_samples[np.newaxis, :, :])
        #
        # # -> [num_range_gates, num_lags, 1]
        # row = samples_for_all_range_lags[..., 1].astype(np.int32)
        #
        # # -> [num_range_gates, num_lags, 1]
        # column = samples_for_all_range_lags[..., 0].astype(np.int32)
        #
        # # -> [num_antennas_1, num_antennas_2, num_range_gates, num_lags]
        # values = correlated[:, :, row, column]
        #
        # return values

        values = []
        if record['lags'].size == 0:
            values.append(np.array([]))
            return values

        num_antennas, num_sequences = samples.shape[:2]
        pulses = list(record['pulses'])
        pulse_phase_offsets = record['pulse_phase_offset']
        ppo_flag = False
        if len(pulse_phase_offsets) != len(record['pulses']):
            if len(pulse_phase_offsets) != 0:
                if not np.isnan(pulse_phase_offsets[0]):
                    pulse_phase_offsets = pulse_phase_offsets.reshape((num_sequences, len(record['pulses'])))
                    ppo_flag = True

        # First range offset in samples
        sample_off = record['first_range_rtt'] * 1e-6 * record['rx_sample_rate']
        sample_off = np.int32(sample_off)

        # Helpful values converted to units of samples
        range_off = np.arange(record['num_ranges'], dtype=np.int32) + sample_off
        tau_in_samples = record['tau_spacing'] * 1e-6 * record['rx_sample_rate']
        lag_pulses_as_samples = np.array(record['lags'], np.int32) * np.int32(tau_in_samples)

        num_range_gates = record['num_ranges']
        num_lags = lag_pulses_as_samples.shape[0]

        # [num_range_gates, 1, 1]
        # [1, num_lags, 2]
        samples_for_all_range_lags = (range_off[..., np.newaxis, np.newaxis] +
                                      lag_pulses_as_samples[np.newaxis, :, :])

        # [num_range_gates, num_lags, 2]
        row = samples_for_all_range_lags[..., 1].astype(np.int32)

        # [num_range_gates, num_lags, 2]
        column = samples_for_all_range_lags[..., 0].astype(np.int32)

        samples = np.swapaxes(samples, 0, 1)    # swap num_antennas, num_sequences

        # [num_sequences, num_antennas, num_range_gates, num_lags]
        values = np.zeros(samples.shape[:2] + row.shape[:2], dtype=np.complex64)

        # Find the correlations
        for lag in range(row.shape[1]):
            values[..., lag] = np.einsum('sij,sij->sij', samples[..., row[:, lag]],
                                         samples[..., column[:, lag]].conj())

        # Remove pulse_phase_offsets if they are present
        # TODO: Test that this actually works
        if ppo_flag:
            # The indices in record['pulses'] of the pulses in each lag pair
            # [num_lags]
            lag1_indices = [pulses.index(val) for val in record['lags'][:, 0]]
            lag2_indices = [pulses.index(val) for val in record['lags'][:, 1]]

            # phase offset of first pulse - phase offset of second pulse, for all lag pairs
            # [num_lags]
            angle_offsets = [np.radians(np.float32(pulse_phase_offsets[:, lag1_indices[i]]) -
                                        np.float32(pulse_phase_offsets[:, lag2_indices[i]]))
                             for i in range(len(lag1_indices))]

            # [num_lags]
            phase_offsets = np.exp(1j * np.array(angle_offsets, np.float32))

            values = np.einsum('sijk,sk->sijk', values, phase_offsets)

        # Find the sample that corresponds to the second pulse transmitting
        second_pulse_sample_num = np.int32(tau_in_samples) * record['pulses'][1] - sample_off - 1

        # Replace all ranges which are contaminated by the second pulse for lag 0
        # with the data from those ranges after the final pulse.
        values[..., second_pulse_sample_num:, 0] = values[..., second_pulse_sample_num:, -1]

        # Now correlate antenna to antenna and reshape to the final shape
        # [num_sequences, num_antennas*num_antennas, num_range_gates, num_lags]
        values = np.einsum('sajk,sbjk->sabjk', values, values.conj()).reshape(
            (num_sequences, num_antennas*num_antennas, num_range_gates, num_lags))

        return values
