# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller, Remington Rohel
"""
This file contains functions for converting bfiq files
to rawacf files.
"""
import logging
import numpy as np
from collections import OrderedDict
from typing import Union

from data_processing.convert_base import BaseConvert

postprocessing_logger = logging.getLogger('borealis_postprocessing')


class ProcessBfiq2Rawacf(BaseConvert):
    """
    Class for conversion of Borealis bfiq files into rawacf files. This class inherits from
    BaseConvert, which handles all functionality generic to postprocessing borealis files.

    See Also
    --------
    ConvertFile
    BaseConvert
    ProcessAntennasIQ2Bfiq
    ProcessAntennasIQ2Rawacf

    Attributes
    ----------
    infile: str
        The filename of the input antennas_iq file.
    outfile: str
        The file name of output file
    infile_structure: str
        The write structure of the file. Structures include:
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
                 averaging_method: str = 'mean'):
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
        averaging_method: str
            Method for averaging correlations across sequences. Either 'median' or 'mean'.
        """
        super().__init__(infile, outfile, 'bfiq', 'rawacf', infile_structure, outfile_structure)
        self.averaging_method = averaging_method

        self.process_file()

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
        if averaging_method is None:
            averaging_method = 'mean'
        record['averaging_method'] = averaging_method

        correlations = ProcessBfiq2Rawacf.calculate_correlations(record, averaging_method)
        record['main_acfs'] = correlations[0]
        record['intf_acfs'] = correlations[1]
        record['xcfs'] = correlations[2]

        record['correlation_descriptors'] = ProcessBfiq2Rawacf.get_correlation_descriptors()
        record['correlation_dimensions'] = ProcessBfiq2Rawacf.get_correlation_dimensions(record)
        record = ProcessBfiq2Rawacf.remove_extra_fields(record)

        return record

    @staticmethod
    def calculate_correlations(record: OrderedDict, averaging_method: str) -> tuple:
        """
        Calculates the auto- and cross-correlations for main and interferometer arrays given the bfiq data in record.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing bfiq data and metadata
        averaging_method: str
            Averaging method. Supported types are 'mean' and 'median'

        Returns
        -------
        main_acfs: np.array
            Autocorrelation of the main array data
        intf_acfs: np.array
            Autocorrelation of the interferometer array data
        xcfs: np.array
            Cross-correlation of the main and interferometer arrays
        """
        # bfiq data shape  = [num_arrays, num_sequences, num_beams, num_samps]
        bfiq_data = record['data']

        # Get the data and reshape
        num_arrays, num_sequences, num_beams, num_samps = record['data_dimensions']
        bfiq_data = bfiq_data.reshape(record['data_dimensions'])

        main_corrs_unavg = ProcessBfiq2Rawacf.correlations_from_samples(bfiq_data[0, ...],
                                                                        bfiq_data[0, ...],
                                                                        record)
        intf_corrs_unavg = ProcessBfiq2Rawacf.correlations_from_samples(bfiq_data[1, ...],
                                                                        bfiq_data[1, ...],
                                                                        record)
        cross_corrs_unavg = ProcessBfiq2Rawacf.correlations_from_samples(bfiq_data[1, ...],
                                                                         bfiq_data[0, ...],
                                                                         record)

        if averaging_method == 'median':
            main_corrs = np.median(np.real(main_corrs_unavg), axis=0) + 1j * np.median(np.imag(main_corrs_unavg),
                                                                                       axis=0)
            intf_corrs = np.median(np.real(intf_corrs_unavg), axis=0) + 1j * np.median(np.imag(intf_corrs_unavg),
                                                                                       axis=0)
            cross_corrs = np.median(np.real(cross_corrs_unavg), axis=0) + 1j * np.median(np.imag(cross_corrs_unavg),
                                                                                         axis=0)
        else:
            # Using mean averaging
            main_corrs = np.einsum('ijkl->jkl', main_corrs_unavg) / num_sequences
            intf_corrs = np.einsum('ijkl->jkl', intf_corrs_unavg) / num_sequences
            cross_corrs = np.einsum('ijkl->jkl', cross_corrs_unavg) / num_sequences

        main_acfs = main_corrs.flatten()
        intf_acfs = intf_corrs.flatten()
        xcfs = cross_corrs.flatten()

        return main_acfs, intf_acfs, xcfs

    @staticmethod
    def correlations_from_samples(beamformed_samples_1: np.array, beamformed_samples_2: np.array,
                                  record: OrderedDict) -> np.array:
        """
        Correlate two sets of beamformed samples together. Correlation matrices are used and
        indices corresponding to lag pulse pairs are extracted.

        Parameters
        ----------
        beamformed_samples_1: ndarray [num_sequences, num_beams, num_samples]
            The first beamformed samples.
        beamformed_samples_2: ndarray [num_sequences, num_beams, num_samples]
            The second beamformed samples.
        record: OrderedDict
            hdf5 record containing bfiq data and metadata

        Returns
        -------
        values: np.array [num_sequences, num_beams, num_ranges, num_lags]
            Array of correlations for each sequence, beam, range, and lag
        """

        # beamformed_samples_1: [num_sequences, num_beams, num_samples]
        # beamformed_samples_2: [num_sequences, num_beams, num_samples]
        # correlated:           [num_sequences, num_beams, num_samples, num_samples]
        correlated = np.einsum('ijk,ijl->ijkl', beamformed_samples_1, beamformed_samples_2.conj())


        values = []
        if record['lags'].size == 0:
            values.append(np.array([]))
            return values

        num_sequences = beamformed_samples_1.shape[0]
        pulses = list(record['pulses'])
        pulse_phase_offsets = record['pulse_phase_offset']
        if len(pulse_phase_offsets) != len(record['pulses']):
            if len(pulse_phase_offsets) != 0:
                pulse_phase_offsets = pulse_phase_offsets.reshape((num_sequences, len(record['pulses'])))

        # First range offset in samples
        sample_off = record['first_range_rtt'] * 1e-6 * record['rx_sample_rate']
        sample_off = np.int32(sample_off)

        # Helpful values converted to units of samples
        range_off = np.arange(record['num_ranges'], dtype=np.int32) + sample_off
        tau_in_samples = record['tau_spacing'] * 1e-6 * record['rx_sample_rate']
        lag_pulses_as_samples = np.array(record['lags'], np.int32) * np.int32(tau_in_samples)

        # [num_range_gates, 1, 1]
        # [1, num_lags, 2]
        samples_for_all_range_lags = (range_off[..., np.newaxis, np.newaxis] +
                                      lag_pulses_as_samples[np.newaxis, :, :])

        # [num_range_gates, num_lags, 2]
        row = samples_for_all_range_lags[..., 1].astype(np.int32)

        # [num_range_gates, num_lags, 2]
        column = samples_for_all_range_lags[..., 0].astype(np.int32)

        # [num_sequences, num_beams, num_range_gates, num_lags]
        values = correlated[..., row, column]

        # Remove pulse_phase_offsets if they are present
        if len(pulse_phase_offsets) == len(pulses):
            # The indices in record['pulses'] of the pulses in each lag pair
            # [num_lags]
            lag1_indices = [pulses.index(val) for val in record['lags'][:, 0]]
            lag2_indices = [pulses.index(val) for val in record['lags'][:, 1]]

            # phase offset of first pulse - phase offset of second pulse, for all lag pairs
            # [num_lags]
            angle_offsets = [np.radians(np.float32(pulse_phase_offsets[lag1_indices[i]]) -
                                        np.float32(pulse_phase_offsets[lag2_indices[i]]))
                             for i in range(len(lag1_indices))]

            # [num_lags]
            phase_offsets = np.exp(1j * np.array(angle_offsets, np.float32))

            values = np.einsum('ijkl,l->ijkl', values, phase_offsets)
        elif len(pulse_phase_offsets) != 0:
            raise ValueError('Dimensions of pulse_phase_offsets does not match dimensions of pulses')

        # Find the sample that corresponds to the second pulse transmitting
        second_pulse_sample_num = np.int32(tau_in_samples) * record['pulses'][1] - sample_off - 1

        # Replace all ranges which are contaminated by the second pulse for lag 0
        # with the data from those ranges after the final pulse.
        values[..., second_pulse_sample_num:, 0] = values[..., second_pulse_sample_num:, -1]

        return values

    @staticmethod
    def get_correlation_descriptors() -> list:
        """
        Returns a list of descriptors corresponding to correlation data dimensions.
        """
        return ['num_beams', 'num_ranges', 'num_lags']

    @staticmethod
    def get_correlation_dimensions(record: OrderedDict) -> np.array:
        """
        Returns the dimensions of correlation data.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing bfiq data and metadata

        Returns
        -------
        Array of ints characterizing the data dimensions
        """
        return np.array([len(record['beam_azms']), record['num_ranges'], len(record['lags'])], dtype=np.uint32)

    @staticmethod
    def remove_extra_fields(record: OrderedDict) -> OrderedDict:
        """
        Removes fields not needed by the rawacf data format.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing bfiq data and metadata

        Returns
        -------
        record: OrderedDict
            hdf5 record without fields that aren't in the rawacf format
        """
        record.pop('data')
        record.pop('data_descriptors')
        record.pop('data_dimensions')
        record.pop('num_ranges')
        record.pop('num_samps')
        record.pop('pulse_phase_offset')
        record.pop('antenna_arrays_order')

        return record
