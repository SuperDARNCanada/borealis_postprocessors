# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller, Remington Rohel
"""
This file contains functions for converting bfiq files
to rawacf files.
"""
import itertools

import h5py
import logging
import numpy as np
from collections import OrderedDict

from postprocessors import BaseConvert

postprocessing_logger = logging.getLogger('borealis_postprocessing')


class Bfiq2Rawacf(BaseConvert):
    """
    Class for conversion of Borealis bfiq files into rawacf files. This class inherits from
    BaseConvert, which handles all functionality generic to postprocessing borealis files.

    See Also
    --------
    ConvertFile
    BaseConvert

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
    """

    def __init__(self, infile: str, outfile: str, infile_structure: str, outfile_structure: str):
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
        """
        super().__init__(infile, outfile, 'bfiq', 'rawacf', infile_structure, outfile_structure)

    @classmethod
    def process_record(cls, record: OrderedDict, **kwargs) -> OrderedDict:
        """
        Takes a record from a bfiq file and processes it into record for rawacf file.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing bfiq data and metadata

        Returns
        -------
        record: OrderedDict
            record converted to rawacf format
        """
        averaging_method = kwargs.get('averaging_method', 'mean')
        record['averaging_method'] = averaging_method

        correlations = cls.calculate_correlations(record, averaging_method)
        record['main_acfs'] = correlations[0]
        record['intf_acfs'] = correlations[1]
        record['xcfs'] = correlations[2]

        # v0.6.1-xxxxxx -> [0, 6, 1]
        githash = record['borealis_git_hash']
        if isinstance(githash, bytes):
            githash = githash.decode('utf-8')
        version = [int(i) for i in githash.split('-')[0].strip('v').split('.')]
        if version[0] == 0:
            if version[1] < 7:
                record['correlation_descriptors'] = cls.get_correlation_descriptors()
                record['correlation_dimensions'] = cls.get_correlation_dimensions(record)
            else:
                record['data_descriptors'] = np.bytes_(cls.get_correlation_descriptors())
                record['data_dimensions'] = cls.get_correlation_dimensions(record)
        else:
            record['descriptions']['main_acfs'] = "Main array autocorrelations"
            record['units']['main_acfs'] = "a.u. ~ W"
            record['dim_labels']['main_acfs'] = ["beam", "range", "lag"]
            record['dim_scales']['main_acfs'] = [["beam_azms", "beam_nums"], "range_gates", "lag_numbers"]

            record['descriptions']['intf_acfs'] = "Interferometer array autocorrelations"
            record['units']['intf_acfs'] = "a.u. ~ W"
            record['dim_labels']['intf_acfs'] = ["beam", "range", "lag"]
            record['dim_scales']['intf_acfs'] = [["beam_azms", "beam_nums"], "range_gates", "lag_numbers"]

            record['descriptions']['xcfs'] = "Cross-correlations between main and interferometer arrays"
            record['units']['xcfs'] = "a.u. ~ W"
            record['dim_labels']['xcfs'] = ["beam", "range", "lag"]
            record['dim_scales']['xcfs'] = [["beam_azms", "beam_nums"], "range_gates", "lag_numbers"]

        record = cls.remove_extra_fields(record)

        return record

    @classmethod
    def _update_metadata(cls, record: OrderedDict, metadata: h5py.Group, **kwargs):
        """
        Adds in metadata fields required for this file type.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing one averaging period worth of data and metadata
        metadata: h5py.Group
            metadata group in the output file

        """
        averaging_method = kwargs.get("averaging_method", "mean")
        dset = metadata.create_dataset("averaging_method", data=averaging_method)
        dset.attrs["description"] = "Averaging method, e.g. mean, median"

        if "lags" not in metadata.keys():
            lag_pulse_table = cls.create_lag_table(record)
            dset = metadata.create_dataset("lags", data=np.arange(len(lag_pulse_table)))
            dset.attrs["description"] = "Lag indices"
            dset.make_scale("lag")

            dset = metadata.create_dataset("lag_numbers", data=lag_pulse_table[1] - lag_pulse_table[0])
            dset.attrs["description"] = "Difference in units of tau_spacing of unique pairs of pulse in the pulse array"
            dset.attrs["units"] = "tau_spacing"
            dset.make_scale("lag")

            dset = metadata.create_dataset("lag_pulse_descriptors",
                                           data=np.array([b'first pulse', b'second pulse']))
            dset.attrs["description"] = "Descriptor of the pulse pairs used in a lag"
            dset.make_scale()

            dset = metadata.create_dataset("lag_pulses", data=lag_pulse_table)
            dset.attrs["description"] = "Unique pairs of pulses in pulse array, in units of tau_spacing"
            dset.attrs["units"] = "tau_spacing"
            dset.dims[0].label = "lag"
            dset.dims[1].label = "pulse"
            dset.dims[0].attach_scale(metadata["lags"])
            dset.dims[1].attach_scale(metadata["lag_pulse_descriptors"])

    @classmethod
    def calculate_correlations(cls, record: OrderedDict, averaging_method: str) -> tuple:
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
        if "descriptions" in record:
            bfiq_data = record["bfiq_data"]
            num_arrays, num_sequences, num_beams, num_samps = bfiq_data.shape
        else:
            bfiq_data = record['data']
            num_arrays, num_sequences, num_beams, num_samps = record['data_dimensions']
            bfiq_data = bfiq_data.reshape(record['data_dimensions'])

        main_corrs_unavg = cls.correlations_from_samples(bfiq_data[0, ...], bfiq_data[0, ...], record)
        intf_corrs_unavg = cls.correlations_from_samples(bfiq_data[1, ...], bfiq_data[1, ...], record)
        cross_corrs_unavg = cls.correlations_from_samples(bfiq_data[1, ...], bfiq_data[0, ...], record)

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

        main_acfs = main_corrs
        intf_acfs = intf_corrs
        xcfs = cross_corrs

        return main_acfs, intf_acfs, xcfs

    @classmethod
    def correlations_from_samples(cls, beamformed_samples_1: np.array, beamformed_samples_2: np.array,
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

        values = []
        if record['lags'].size == 0:
            values.append(np.array([]))
            return values

        num_sequences = beamformed_samples_1.shape[0]
        pulses = list(record['pulses'])

        # First range offset in samples
        sample_off = record['first_range_rtt'] * 1e-6 * record['rx_sample_rate']
        sample_off = np.int32(sample_off)

        # Helpful values converted to units of samples
        tau_in_samples = record['tau_spacing'] * 1e-6 * record['rx_sample_rate']
        if "range_gates" in record:
            range_off = record["range_gates"] + sample_off
            lag_pulses_as_samples = record['lag_pulses'] * np.int32(tau_in_samples)
        else:
            range_off = np.arange(record['num_ranges'], dtype=np.int32) + sample_off
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
        values = np.zeros(beamformed_samples_1.shape[:2] + row.shape[:2], dtype=np.complex64)

        # Find the correlations
        for lag in range(row.shape[1]):
            values[..., lag] = beamformed_samples_1[..., row[:, lag]] * beamformed_samples_2[..., column[:, lag]].conj()

        if "pulse_phase_offset" in record:
            pulse_phase_offsets = record['pulse_phase_offset']
            ppo_flag = False
            if len(pulse_phase_offsets) != len(record['pulses']):
                if len(pulse_phase_offsets) > 1:
                    if not np.isnan(pulse_phase_offsets[0]):
                        pulse_phase_offsets = pulse_phase_offsets.reshape((num_sequences, len(record['pulses'])))
                        ppo_flag = True

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
            elif len(pulse_phase_offsets) != 0 and ppo_flag:
                raise ValueError('Dimensions of pulse_phase_offsets does not match dimensions of pulses')

        # Find the sample that corresponds to the second pulse transmitting
        second_pulse_sample_num = np.int32(tau_in_samples) * record['pulses'][1] - sample_off - 1

        # Replace all ranges which are contaminated by the second pulse for lag 0
        # with the data from those ranges after the final pulse.
        values[..., second_pulse_sample_num:, 0] = values[..., second_pulse_sample_num:, -1]

        return values

    @classmethod
    def get_correlation_descriptors(cls) -> list:
        """
        Returns a list of descriptors corresponding to correlation data dimensions.
        """
        return ['num_beams', 'num_ranges', 'num_lags']

    @classmethod
    def get_correlation_dimensions(cls, record: OrderedDict) -> np.array:
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

    @classmethod
    def create_lag_table(cls, record: OrderedDict) -> np.array:
        """
        Creates the lag table for the record.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing antennas_iq data and metadata

        Returns
        -------
        lags: np.array
            Array of lag pairs for the record. Each pair is formatted as [0, 1], where
            the first number is the index of the first pulse in units of tau, and the second number
            is the index of the second pulse in units of tau. The lag pairs start with [0, 0], then
            are sorted in ascending order based on difference between the pulses, and finally appended
            with an alternate lag-zero pulse [last_pulse, last_pulse].
        """
        if "lags" in record:
            return record["lags"]

        lag_table = list(itertools.combinations(record['pulses'], 2))  # Create all combinations of lags
        lag_table.append([record['pulses'][0], record['pulses'][0]])  # lag 0
        lag_table = sorted(lag_table, key=lambda x: x[1] - x[0])  # sort by lag number
        lag_table.append([record['pulses'][-1], record['pulses'][-1]])  # alternate lag 0
        lags = np.array(lag_table, dtype=np.uint32)

        return lags

    @classmethod
    def remove_extra_fields(cls, record: OrderedDict) -> OrderedDict:
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
        def remove_field(name):
            """Removes a field completely from a Borealis v1.0+ record"""
            record.pop(name)
            if name in record["descriptions"]:
                record["descriptions"].pop(name)
            if name in record["units"]:
                record["units"].pop(name)
            if name in record["dim_labels"]:
                record["dim_labels"].pop(name)
            if name in record["dim_scales"]:
                record["dim_scales"].pop(name)
            if name in record["dim_nicknames"]:
                record["dim_nicknames"].pop(name)

        if "descriptions" in record:
            remove_field("antenna_arrays")
            remove_field("bfiq_data")
            if "pulse_phase_offset" in record:
                remove_field("pulse_phase_offset")
            remove_field("sample_time")
        else:
            githash = record['borealis_git_hash'].split('-')[0].strip('v').split('.')  # v0.6.1-xxxxxx -> ['0', '6', '1']
            if githash[0] == '0' and int(githash[1]) < 7:
                record.pop('data_descriptors')
                record.pop('data_dimensions')
            record.pop('data')
            record.pop('num_ranges')
            record.pop('num_samps')
            record.pop('pulse_phase_offset')
            record.pop('antenna_arrays_order')

        return record
