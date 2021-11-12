# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller, Remington Rohel
"""
This file contains functions for converting antennas_iq files
to bfiq files.
"""
import itertools
from collections import OrderedDict
from typing import Union

import numpy as np
from scipy.constants import speed_of_light

from data_processing.convert_base import BaseConvert

try:
    import cupy as xp
except ImportError:
    import numpy as xp

    cupy_available = False
else:
    cupy_available = True

import logging

postprocessing_logger = logging.getLogger('borealis_postprocessing')


class ProcessAntennasIQ2Bfiq(BaseConvert):
    """
    Class for conversion of Borealis antennas_iq files into bfiq files. This class inherits from
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
        The write structure of the file. Structures include:
        'array'
        'site'
    outfile_structure: str
        The desired structure of the output file. Same structures as above, plus 'iqdat'.
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
            Borealis structure of output file. Either 'array', 'site', or 'iqdat'.
        """
        super().__init__(infile, outfile, 'antennas_iq', 'bfiq', infile_structure, outfile_structure)

        self.process_file()

    @staticmethod
    def process_record(record: OrderedDict, averaging_method: Union[None, str]) -> OrderedDict:
        """
        Takes a record from an antennas_iq file and converts it into a bfiq record.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing antennas_iq data and metadata
        averaging_method: Union[None, str]
            Method to use for averaging correlations across sequences. Unused by this method.

        Returns
        -------
        record: OrderedDict
            hdf5 record, with new fields required by bfiq data format
        """
        record['first_range'] = ProcessAntennasIQ2Bfiq.calculate_first_range(record)
        record['first_range_rtt'] = ProcessAntennasIQ2Bfiq.calculate_first_range_rtt(record)
        record['lags'] = ProcessAntennasIQ2Bfiq.create_lag_table(record)
        record['range_sep'] = ProcessAntennasIQ2Bfiq.calculate_range_separation(record)
        record['num_ranges'] = ProcessAntennasIQ2Bfiq.get_number_of_ranges(record)
        record['data'] = ProcessAntennasIQ2Bfiq.beamform_data(record)
        record['data_descriptors'] = ProcessAntennasIQ2Bfiq.get_data_descriptors()
        record['data_dimensions'] = ProcessAntennasIQ2Bfiq.get_data_dimensions(record)
        record['antenna_arrays_order'] = ProcessAntennasIQ2Bfiq.change_antenna_arrays_order()

        return record

    @staticmethod
    def beamform_data(record: OrderedDict) -> np.array:
        """
        Beamforms the data for each array, and stores it back into the record

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing antennas_iq data and metadata

        Returns
        -------
        all_data: np.array
            Array containing all data after beamforming, and grouped by antenna array.
        """
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

        main_beamformed_data = xp.array([], dtype=xp.complex64)
        intf_beamformed_data = xp.array([], dtype=xp.complex64)
        main_antenna_count = record['main_antenna_count']

        station = record['station']
        main_antenna_spacing = radar_dict[station]['main_antenna_spacing']
        intf_antenna_spacing = radar_dict[station]['intf_antenna_spacing']

        # Loop through every sequence and beamform the data.
        # Output shape after loop is [num_sequences, num_beams, num_samps]
        for sequence in range(num_sequences):
            # data input shape  = [num_antennas, num_samps]
            # data return shape = [num_beams, num_samps]
            main_beamformed_data = \
                xp.append(main_beamformed_data,
                          ProcessAntennasIQ2Bfiq.beamform(antennas_data[:main_antenna_count, sequence, :],
                                                          beam_azms,
                                                          freq,
                                                          main_antenna_spacing))
            intf_beamformed_data = \
                xp.append(intf_beamformed_data,
                          ProcessAntennasIQ2Bfiq.beamform(antennas_data[main_antenna_count:, sequence, :],
                                                          beam_azms,
                                                          freq,
                                                          intf_antenna_spacing))

        all_data = xp.append(main_beamformed_data, intf_beamformed_data).flatten()

        return all_data

    @staticmethod
    def beamform(antennas_data: np.array, beamdirs: np.array, rxfreq: float, antenna_spacing: float) -> np.array:
        """
        Beamforms the data from each antenna and sums to create one dataset for each beam direction.

        Parameters
        ----------
        antennas_data: np.array
            Numpy array of dimensions num_antennas x num_samps. All antennas are assumed to be from the same array
            and are assumed to be side by side with uniform antenna spacing
        beamdirs: np.array
            Azimuthal beam directions in degrees off boresight
        rxfreq: float
            Frequency of the received beam
        antenna_spacing: float
            Spacing in metres between antennas (assumed uniform)

        Returns
        -------
        beamformed_data: np.array
            Array of shape [num_beams, num_samps]
        """
        beamformed_data = []

        # [num_antennas, num_samps]
        num_antennas, num_samps = antennas_data.shape

        # Loop through all beam directions
        for beam_direction in beamdirs:
            antenna_phase_shifts = []

            # Get phase shift for each antenna
            for antenna in range(num_antennas):
                phase_shift = ProcessAntennasIQ2Bfiq.get_phshift(beam_direction,
                                                                 rxfreq,
                                                                 antenna,
                                                                 num_antennas,
                                                                 antenna_spacing)
                # Bring into range (-2*pi, 2*pi)
                phase_shift = xp.fmod(phase_shift, 2 * xp.pi)
                antenna_phase_shifts.append(phase_shift)

            # Apply phase shift to data from respective antenna
            phased_antenna_data = [ProcessAntennasIQ2Bfiq.shift_samples(antennas_data[i], antenna_phase_shifts[i], 1.0)
                                   for i in range(num_antennas)]
            phased_antenna_data = xp.array(phased_antenna_data)

            # Sum across antennas to get beamformed data
            one_beam_data = xp.sum(phased_antenna_data, axis=0)
            beamformed_data.append(one_beam_data)
        beamformed_data = xp.array(beamformed_data)

        return beamformed_data

    @staticmethod
    def get_phshift(beamdir: float, freq: float, antenna: int, num_antennas: int, antenna_spacing: float,
                    centre_offset: int = 0.0) -> float:
        """
        Find the phase shift for a given antenna and beam direction.
        Form the beam given the beam direction (degrees off boresite), the tx frequency, the antenna number,
        a specified extra phase shift if there is any, the number of antennas in the array, and the spacing
        between antennas.

        Parameters
        ----------
        beamdir: float
            The azimuthal direction of the beam off boresight, in degrees, positive beamdir being to
            the right of the boresight (looking along boresight from ground). This is for this antenna.
        freq: float
            Transmit frequency in kHz
        antenna: int
            Antenna number, INDEXED FROM ZERO, zero being the leftmost antenna if looking down the boresight
            and positive beamdir right of boresight
        num_antennas: int
            Number of antennas in this array
        antenna_spacing: float
            Distance between antennas in this array, in meters
        centre_offset: float
            The phase reference for the midpoint of the array. Default = 0.0, in metres.
            Important if there is a shift in centre point between arrays in the direction along the array.
            Positive is shifted to the right when looking along boresight (from the ground).

        Returns
        -------
        phshift: float
            A phase shift for the samples for this antenna number, in radians.
        """
        freq = freq * 1000.0  # convert to Hz.

        # Convert to radians
        beamrad = xp.pi * xp.float64(beamdir) / 180.0

        # Pointing to right of boresight, use point in middle (hypothetically antenna 7.5) as phshift=0
        phshift = 2 * xp.pi * freq * (((num_antennas - 1) / 2.0 - antenna) * antenna_spacing + centre_offset) * \
                  xp.cos(xp.pi / 2.0 - beamrad) / speed_of_light

        # Bring into range (-2*pi, 2*pi)
        phshift = xp.fmod(phshift, 2 * xp.pi)

        return phshift

    @staticmethod
    def shift_samples(basic_samples: np.array, phshift: float, amplitude: float = 1.) -> np.array:
        """
        Shift samples for a pulse by a given phase shift.
        Take the samples and shift by given phase shift in rads and adjust amplitude as
        required for imaging.

        Parameters
        ----------
        basic_samples: np.array
            Samples for this pulse
        phshift: float
            phase for this antenna to offset by in rads
        amplitude: float
            Amplitude for this antenna (= 1 if not imaging)

        Returns
        -------
        samples: np.array
            Basic_samples that have been shaped for the antenna for the desired beam.
        """
        samples = amplitude * xp.exp(1j * phshift) * basic_samples

        return samples

    @staticmethod
    def calculate_first_range(record: OrderedDict) -> float:
        """
        Calculates the distance from the main array to the first range (in km).

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing antennas_iq data and metadata

        Returns
        -------
        first_range: float
            Distance to first range in km
        """
        # TODO: Get this from somewhere, probably linked to the experiment ran. Might need to look up
        #   based on githash
        first_range = 180.0  # scf.FIRST_RANGE

        return xp.float32(first_range)

    @staticmethod
    def calculate_first_range_rtt(record: OrderedDict) -> float:
        """
        Calculates the round-trip time (in microseconds) to the first range in a record.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing antennas_iq data and metadata

        Returns
        -------
        first_range_rtt: float
            Time that it takes signal to travel to first range gate and back, in microseconds
        """
        # km * (there and back) * (km to meters) * (seconds to us) / c
        first_range_rtt = record['first_range'] * 2.0 * 1.0e3 * 1e6 / speed_of_light

        return xp.float32(first_range_rtt)

    @staticmethod
    def create_lag_table(record: OrderedDict) -> np.array:
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
        lag_table = list(itertools.combinations(record['pulses'], 2))   # Create all combinations of lags
        lag_table.append([record['pulses'][0], record['pulses'][0]])    # lag 0
        lag_table = sorted(lag_table, key=lambda x: x[1] - x[0])        # sort by lag number
        lag_table.append([record['pulses'][-1], record['pulses'][-1]])  # alternate lag 0
        lags = xp.array(lag_table, dtype=xp.uint32)

        return lags

    @staticmethod
    def calculate_range_separation(record: OrderedDict) -> float:
        """
        Calculates the separation between ranges in km.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing antennas_iq data and metadata

        Returns
        -------
        range_sep: float
            The separation between adjacent ranges, in km.
        """
        # (1 / (sample rate)) * c / (km to meters) / 2
        range_sep = 1 / record['rx_sample_rate'] * speed_of_light / 1.0e3 / 2.0

        return xp.float32(range_sep)

    @staticmethod
    def get_number_of_ranges(record: OrderedDict) -> int:
        """
        Gets the number of ranges for the record.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing antennas_iq data and metadata

        Returns
        -------
        num_ranges: int
            The number of ranges of the data
        """
        # Infer the number of ranges from the record metadata
        first_range_offset = ProcessAntennasIQ2Bfiq.calculate_first_range_rtt(record) * 1e-6 * record['rx_sample_rate']
        num_ranges = record['num_samps'] - xp.int32(first_range_offset) - record['blanked_samples'][-1]

        # 3 extra samples taken for each record (not sure why)
        num_ranges = num_ranges - 3

        return num_ranges

    @staticmethod
    def get_data_descriptors() -> list:
        """
        Returns the proper data descriptors for a bfiq file

        Returns
        -------
        new_descriptors: list
            List of descriptors for data dimensions of bfiq file
        """
        new_descriptors = ['num_antenna_arrays', 'num_sequences', 'num_beams', 'num_samps']

        return new_descriptors

    @staticmethod
    def get_data_dimensions(record: OrderedDict):
        """
        Returns a list of the new data dimensions for a bfiq record.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing antennas_iq data and metadata

        Returns
        -------
        new_dimensions: np.array
            Dimensions of data in bfiq record
        """
        # Old dimensions: [num_antennas, num_sequences, num_samps]
        # New dimensions: [num_antenna_arrays, num_sequences, num_beams, num_samps]
        old_dimensions = record['data_dimensions']

        new_dimensions = xp.array([2, old_dimensions[1], len(record['beam_azms']), old_dimensions[2]],
                                  dtype=xp.uint32)

        return new_dimensions

    @staticmethod
    def change_antenna_arrays_order() -> list:
        """
        Returns the correct field 'antenna_arrays_order' for a bfiq file

        Returns
        -------
        List of array names
        """
        return ['main', 'intf']


radar_dict = {
    'sas': {'main_antenna_spacing': 15.24,
            'intf_antenna_spacing': 15.24},
    'pgr': {'main_antenna_spacing': 15.24,
            'intf_antenna_spacing': 15.24},
    'cly': {'main_antenna_spacing': 15.24,
            'intf_antenna_spacing': 15.24},
    'rkn': {'main_antenna_spacing': 15.24,
            'intf_antenna_spacing': 15.24},
    'inv': {'main_antenna_spacing': 15.24,
            'intf_antenna_spacing': 15.24}
}
