# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller, Remington Rohel
"""
This file contains functions for converting antennas_iq files
to bfiq files.
"""
import itertools
from collections import OrderedDict

import h5py
import numpy as np
from scipy.constants import speed_of_light

from postprocessors import BaseConvert

import logging

postprocessing_logger = logging.getLogger('borealis_postprocessing')


class AntennasIQ2Bfiq(BaseConvert):
    """
    Class for conversion of Borealis antennas_iq files into bfiq files. This class inherits from
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
        The desired structure of the output file. Same structures as above, plus 'iqdat'.
    """
    window = [1.0] * 16

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

    @classmethod
    def process_record(cls, record: OrderedDict, **kwargs) -> OrderedDict:
        """
        Takes a record from an antennas_iq file and converts it into a bfiq record.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing antennas_iq data and metadata

        Returns
        -------
        record: OrderedDict
            hdf5 record, with new fields required by bfiq data format
        """
        githash = record['borealis_git_hash']
        if isinstance(githash, bytes):
            githash = githash.decode('utf-8')
        borealis_major_version = int(githash.split('-')[0].lstrip('v').split('.')[0])
        if borealis_major_version < 1:
            record['first_range'] = cls.calculate_first_range(record)
            record['first_range_rtt'] = cls.calculate_first_range_rtt(record)
            record['lags'] = cls.create_lag_table(record)
            record['range_sep'] = cls.calculate_range_separation(record)
            record['num_ranges'] = cls.get_number_of_ranges(record)
            record['data'] = cls.beamform_data(record)
            record['data_descriptors'] = cls.get_data_descriptors()
            record['data_dimensions'] = cls.get_data_dimensions(record)
            record['antenna_arrays_order'] = cls.change_antenna_arrays_order()
        else:
            record['antenna_arrays'] = cls.add_antenna_arrays(record)
            record['bfiq_data'] = cls.beamform_data(record)
            record['descriptions']['bfiq_data'] = "Beamformed I&Q complex voltage samples for each antenna array"
            record['units']['bfiq_data'] = "a.u. ~ V"
            record['dim_labels']['bfiq_data'] = ["array", "sequence", "beam", "time"]
            record['dim_scales']['bfiq_data'] = [
                ["antenna_arrays"],
                ["sqn_timestamps"],
                ["beam_nums", "beam_azms"],
                ["sample_time"],
            ]
            del record['antennas_iq_data']

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
        kwargs: dict
            unused by this class
        """
        dset = metadata.create_dataset("antenna_arrays", data=cls.add_antenna_arrays(record))
        dset.attrs["description"] = "Descriptor of each antenna array contained in the data"

    @classmethod
    def beamform_data(cls, record: OrderedDict) -> np.array:
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
        station = record['station']

        if 'antennas_iq_data' in record.keys():  # Borealis v1.0+ record
            antennas_data = record['antennas_iq_data']
            num_antennas, num_sequences, num_samps = antennas_data.shape
            main_indices = [i for i in record['rx_antennas'] if i in record['rx_main_antennas']]
            intf_indices = [i for i in record['rx_antennas'] if i in record['rx_intf_antennas']]
            main_data = antennas_data[main_indices, ...]
            intf_data = antennas_data[intf_indices, ...]
            main_locations = record['antenna_locations'][main_indices][:, 0]  # only the x coordinate
            intf_locations = record['antenna_locations'][intf_indices][:, 0]
        else:
            antennas_data = record['data']
            num_antennas, num_sequences, num_samps = record['data_dimensions']
            antennas_data = antennas_data.reshape(record['data_dimensions'])
            main_antenna_count = record['main_antenna_count']
            intf_antenna_count = record['intf_antenna_count']
            main_antenna_spacing = radar_dict[station]['main_antenna_spacing']
            intf_antenna_spacing = radar_dict[station]['intf_antenna_spacing']
            intf_offset = radar_dict[station]['intf_x_offset']
            antenna_indices = np.array([int(i.split('_')[-1]) for i in record['antenna_arrays_order']])
            main_antenna_indices = np.array([i for i in antenna_indices if i < main_antenna_count])
            intf_antenna_indices = np.array([i - main_antenna_count for i in antenna_indices if i >= main_antenna_count])

            # Middle of array is origin, boresight is in +y direction, so antenna=0 is the most negative x-coordinate
            main_locations = np.array([(ant - (main_antenna_count - 1) / 2.0) * main_antenna_spacing
                                       for ant in main_antenna_indices])
            intf_locations = np.array([(ant - (intf_antenna_count - 1) / 2.0) * intf_antenna_spacing + intf_offset
                                       for ant in intf_antenna_indices])

            main_data = antennas_data[:len(main_antenna_indices), ...]
            intf_data = antennas_data[len(main_antenna_indices):, ...]

        # Loop through every sequence and beamform the data.
        # Output shape after loop is [num_sequences, num_beams, num_samps]
        main_beamformed_data = []
        intf_beamformed_data = []
        for sequence in range(num_sequences):
            # data input shape  = [num_antennas, num_samps]
            # data return shape = [num_beams, num_samps]
            main_beamformed_data.append(cls.beamform(main_data[:, sequence, :], beam_azms, freq, main_locations))

            intf_beamformed_data.append(cls.beamform(intf_data[:, sequence, :], beam_azms, freq, intf_locations))

        main_data = np.array(main_beamformed_data)
        intf_data = np.array(intf_beamformed_data)
        all_data = np.stack((main_data, intf_data), axis=0)

        return all_data

    @classmethod
    def beamform(cls, antennas_data: np.array, beamdirs: np.array, rxfreq: float, locations: np.array) -> np.array:
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
        locations: np.array
            x-positions of each antenna in the array

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
            antenna_phase_shifts = cls.get_phshift(beam_direction, rxfreq, locations)

            # Apply phase shift to data from respective antenna
            if num_antennas == 16:
                phased_antenna_data = [cls.shift_samples(antennas_data[i], antenna_phase_shifts[i], cls.window[i])
                                       for i in range(num_antennas)]
            else:
                phased_antenna_data = [cls.shift_samples(antennas_data[i], antenna_phase_shifts[i], 1.0)
                                       for i in range(num_antennas)]
            phased_antenna_data = np.array(phased_antenna_data)

            # Sum across antennas to get beamformed data
            one_beam_data = np.sum(phased_antenna_data, axis=0)
            beamformed_data.append(one_beam_data)
        beamformed_data = np.array(beamformed_data)

        return beamformed_data

    @classmethod
    def get_phshift(cls, beamdir: float, freq_khz: float, locations: np.array) -> float:
        """
        Find the phase shift for a given antenna and beam direction.
        Form the beam given the beam direction (degrees off boresite), the tx frequency, the antenna number,
        a specified extra phase shift if there is any, the number of antennas in the array, and the spacing
        between antennas.

        Parameters
        ----------
        beamdir: float
            The azimuthal direction of the beam off boresight, in degrees, positive beamdir being to
            the right of the boresight (looking along boresight from ground).
        freq_khz: float
            Transmit frequency in kHz
        locations: np.array
            Location (x-coordinate) of each antenna in the array.

        Returns
        -------
        phshift: float
            A phase shift for the samples for this antenna number, in radians.
        """
        freq_hz = freq_khz * 1000.0  # convert to Hz
        k = 2 * np.pi * freq_hz / speed_of_light        # 2pi / wavelength

        # Convert to radians CW of boresight direction
        beamrad = np.deg2rad(beamdir)

        # phshift = 0 at origin
        phshift = -1 * k * locations * np.sin(beamrad)

        return phshift

    @classmethod
    def shift_samples(cls, basic_samples: np.array, phshift: float, amplitude: float = 1.) -> np.array:
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
        samples = amplitude * np.exp(1j * phshift) * basic_samples

        return samples

    @classmethod
    def calculate_first_range(cls, record: OrderedDict) -> float:
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
        return record.get("first_range", np.float32(180.0))

    @classmethod
    def calculate_first_range_rtt(cls, record: OrderedDict) -> float:
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
        return record.get(
            "first_range_rtt",
            np.float32(record['first_range'] * 2.0 * 1.0e3 * 1e6 / speed_of_light)
        )

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

        lag_table = list(itertools.combinations(record['pulses'], 2))   # Create all combinations of lags
        lag_table.append([record['pulses'][0], record['pulses'][0]])    # lag 0
        lag_table = sorted(lag_table, key=lambda x: x[1] - x[0])        # sort by lag number
        lag_table.append([record['pulses'][-1], record['pulses'][-1]])  # alternate lag 0
        lags = np.array(lag_table, dtype=np.uint32)

        return lags

    @classmethod
    def calculate_range_separation(cls, record: OrderedDict) -> float:
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
        return record.get(
            "range_sep",
            np.float32(1 / record['rx_sample_rate'] * speed_of_light / 1.0e3 / 2.0)
        )

    @classmethod
    def get_number_of_ranges(cls, record: OrderedDict) -> int:
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
        if "num_ranges" in record.keys():
            return record["num_ranges"]

        # Infer the number of ranges from the record metadata
        first_range_offset = cls.calculate_first_range_rtt(record) * 1e-6 * record['rx_sample_rate']
        tau_in_samples = int(round(record['tau_spacing'] * 1e-6 * record['rx_sample_rate']))
        num_ranges = record['num_samps'] - np.int32(first_range_offset) - (record['pulses'][-1] * tau_in_samples)

        # 3 extra samples taken for each record (not sure why)
        num_ranges = num_ranges - 3

        return np.uint32(num_ranges)

    @classmethod
    def get_data_descriptors(cls) -> list:
        """
        Returns the proper data descriptors for a bfiq file

        Returns
        -------
        new_descriptors: list
            List of descriptors for data dimensions of bfiq file
        """
        new_descriptors = ['num_antenna_arrays', 'num_sequences', 'num_beams', 'num_samps']

        return new_descriptors

    @classmethod
    def get_data_dimensions(cls, record: OrderedDict):
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

        new_dimensions = np.array([2, old_dimensions[1], len(record['beam_azms']), old_dimensions[2]],
                                  dtype=np.uint32)

        return new_dimensions

    @classmethod
    def change_antenna_arrays_order(cls) -> np.array:
        """
        Returns the correct field 'antenna_arrays_order' for a bfiq file

        Returns
        -------
        List of array names
        """
        return ['main', 'intf']

    @classmethod
    def add_antenna_arrays(cls, record: OrderedDict):
        """
        Returns the correct field 'antenna_arrays' for a borealis v1.0+ bfiq file.

        Returns
        -------
        List of array names
        """
        arrays = ['main']
        if 'rx_intf_antennas' in record.keys() and len(record['rx_intf_antennas']) > 0:
            arrays.append('intf')
        return arrays


radar_dict = {
    'sas': {'main_antenna_spacing': 15.24,
            'intf_antenna_spacing': 15.24,
            'intf_x_offset': 0.0},
    'pgr': {'main_antenna_spacing': 15.24,
            'intf_antenna_spacing': 15.24,
            'intf_x_offset': 0.0},
    'cly': {'main_antenna_spacing': 15.24,
            'intf_antenna_spacing': 15.24,
            'intf_x_offset': 0.0},
    'rkn': {'main_antenna_spacing': 15.24,
            'intf_antenna_spacing': 15.24,
            'intf_x_offset': 0.0},
    'inv': {'main_antenna_spacing': 15.24,
            'intf_antenna_spacing': 15.24,
            'intf_x_offset': 1.5},
    'lab': {'main_antenna_spacing': 15.24,
            'intf_antenna_spacing': 15.24,
            'intf_x_offset': 0.0},
    'wal': {'main_antenna_spacing': 12.8016,
            'intf_antenna_spacing': 12.8016,
            'intf_x_offset': 0.0}
}
