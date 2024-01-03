# Copyright 2023 SuperDARN Canada, University of Saskatchewan

"""
This file contains functions for imaging antennas_iq files to rawacf files,
using Singular Value Decomposition to solve the imaging inverse problem.
"""
from collections import OrderedDict
import numpy as np
import h5py

from postprocessors import AntennasIQ2Rawacf, AntennasIQ2Bfiq


class SVDImaging(AntennasIQ2Rawacf):
    """
    Class for conversion of Borealis antennas_iq files into rawacf files using SVD imaging. This class
    inherits from BaseConvert, which handles all functionality generic to postprocessing borealis files.

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
    """

    def __init__(self, infile: str, outfile: str):
        """
        Initialize the attributes of the class.

        Parameters
        ----------
        infile: str
            Path to input file.
        outfile: str
            Path to output file.
        """
        super().__init__(infile, outfile, 'site', 'site')

        self.uh, self.s, self.v, self.unique_baselines, self.angles = self.setup_inversion(infile)

        self.process_file()

    @staticmethod
    def setup_inversion(infile: str):
        """
        Calculates the reverse matrix for the inverse problem.

        Parameters
        ----------
        infile: str
            Path to input file.

        Returns
        -------
        """
        with h5py.File(infile, 'r') as f:
            recs = sorted(list(f.keys()))

            # The first record of the file
            record = f[recs[0]]

            # [num_antennas, num_sequences, num_samples]
            data = record['data'][:]
            data_dimensions = record['data_dimensions'][:]
            data = data.reshape(data_dimensions)
            print(data.shape)

            # [x, y] offsets for each antenna
            antenna_locations = np.zeros((data.shape[0], 2), dtype=np.float32)
            antenna_locations[:16, 0] = np.arange(16) * 15.24       # main array, leftmost is at [0, 0]
            antenna_locations[16:, 0] = np.arange(6, 10) * 15.24    # intf array
            antenna_locations[16:, 1] = -100.0                      # intf array 100m behind main array

            # [num_antennas, num_antennas, 2] for [u, v] from antenna_i - antenna_j
            baselines = np.zeros((antenna_locations.shape[0], antenna_locations.shape[0], 2), dtype=np.float32)
            baselines[..., 0] = antenna_locations[:, np.newaxis, 0] - antenna_locations[np.newaxis, :, 0]
            baselines[..., 1] = antenna_locations[:, np.newaxis, 1] - antenna_locations[np.newaxis, :, 1]
            
            # Group all redundant baselines, where (x, y) are keys and indices in baselines are the values
            unique_baselines = OrderedDict()
            for i in range(baselines.shape[0]):
                for j in range(baselines.shape[1]):
                    coords = baselines[i, j].round(decimals=3)
                    if (coords[0], coords[1]) in unique_baselines.keys():
                        unique_baselines[(coords[0], coords[1])].append((i, j))
                    else:
                        unique_baselines[(coords[0], coords[1])] = [(i, j)]
            baselines = np.array(list(unique_baselines.keys()))
            # Result is a dictionary where keys are unique baselines, and values are lists of indices that are redundant

            # Array of directions that we discretize FOV into
            elevations = np.arange(0, 60) * np.pi / 180.0   # in radians
            azimuths = np.arange(-180, 180) * np.pi / 180.0    # in radians

            # Combine azimuths and elevations into single array
            # [num_azimuths, num_elevations, 2]
            angles = np.zeros((azimuths.size, elevations.size, 2), dtype=np.float32)
            angles[:, :, 0] = np.repeat(azimuths[:, np.newaxis], elevations.shape[0], axis=1)
            angles[:, :, 1] = np.repeat(elevations[np.newaxis, :], azimuths.shape[0], axis=0)

            # directions array is [num_directions, 2] of local [x, y] coordinates of look direction at radar site
            directions = np.zeros((elevations.size * azimuths.size, 2), dtype=np.float32)
            directions[:, 0] = np.einsum('i,j->ij', np.sin(azimuths), np.cos(elevations)).reshape(directions.shape[0])
            directions[:, 1] = np.einsum('i,j->ij', np.cos(azimuths), np.cos(elevations)).reshape(directions.shape[0])
            
            # Calculate dot product of look direction with baseline
            # -> [num_unique_baselines, num_directions]
            projections = np.einsum('xi,yi->xy', baselines, directions)     # dot product of direction with baseline

            # Convert from distance to phase difference
            freq = record.attrs['freq'] * 1000        # freq is stored in kHz
            k = 2 * np.pi * freq / 299792458    # wavenumber
            phase_differences = k * projections

            # Calculate exp(-j * phase_differences), the forward matrix between brightness and visibilities
            # -> [num_unique_baselines, num_brightnesses]
            forward_matrix = np.exp(-1j * phase_differences).reshape((len(unique_baselines.keys()), -1))

            # Calculate the SVD of forward_matrix
            u, s, vh = np.linalg.svd(forward_matrix, full_matrices=False)

            return u.T.conj(), s, vh.T.conj(), unique_baselines, angles

    def process_record(self, record: OrderedDict, **kwargs) -> OrderedDict:
        """
        Takes a record from an antennas_iq file process into a rawacf record.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing antennas_iq data and metadata

        Returns
        -------
        record: OrderedDict
            hdf5 record, with new fields required by rawacf data format
        """
        # [num_antennas, num_sequences, num_samples]
        data = record['data'][:]
        data = data.reshape(record['data_dimensions'][:])
        num_antennas, num_sequences, num_samples = record['data_dimensions'][:]

        # Location of pulses, in units of tau from start of samples
        pulse_table = record['pulses']
        num_pulses = len(pulse_table)

        # Convert tau into units of samples
        tau_in_samples = np.int32(record['tau_spacing'] * 1e-6 * record['rx_sample_rate'])

        # Convert the pulses to units of samples
        pulses_in_samples = pulse_table * tau_in_samples

        # Offset from pulse to first range, in samples
        first_range = AntennasIQ2Bfiq.calculate_first_range(record)
        first_range_rtt = first_range * 2.0 * 1.0e3 * 1e6 / 299792458
        sample_offset = np.int32(first_range_rtt * 1e-6 * record['rx_sample_rate'])

        # Range gates
        # Infer the number of ranges from the record metadata
        num_ranges = record['num_samps'] - np.int32(sample_offset) - record['blanked_samples'][-1]
        # 3 extra samples taken for each record (not sure why)
        num_ranges = num_ranges - 3
        ranges = np.arange(num_ranges, dtype=np.int32)

        # This gives the index for each range gate for each pulse, so data at [i, j]
        # is data for range gate i from pulse j
        # [num_pulses, num_ranges]
        range_for_pulses = ranges[np.newaxis, :] + pulses_in_samples[:, np.newaxis] + sample_offset

        # Extract the data from each pulse, for building up ACFs
        # [num_antennas, num_sequences, num_pulses, num_ranges]
        data_for_pulses = np.zeros((num_antennas, num_sequences, num_pulses, num_ranges),
                                   dtype=np.complex64)
        for p in range(len(pulse_table)):
            data_for_pulses[..., p, :] = data[..., range_for_pulses[p, :]]

        # [num_lags, 2] containing indices into pulses which create said lag.
        # Ordered in axis 0 from smallest to largest lag
        lags_in_tau = AntennasIQ2Bfiq.create_lag_table(record).tolist()
        lags = []
        pulses = pulse_table.tolist()
        for lag in lags_in_tau:
            pulse_0_index = pulses.index(lag[0])
            pulse_1_index = pulses.index(lag[1])
            lags.append([pulse_0_index, pulse_1_index])
        lags = np.array(lags)
        num_lags = lags.shape[0]

        # Create all the visibilities by multiplying samples for each antenna
        visibilities = np.zeros((num_antennas, num_antennas, num_sequences, num_lags, num_ranges),
                                dtype=np.complex64)
        for i in range(lags.shape[0]):
            visibilities[..., i, :] = np.einsum('asr,bsr->absr',     # antenna sequence range x antenna sequence range
                                                data_for_pulses[..., lags[i, 0], :],
                                                data_for_pulses[..., lags[i, 1], :].conj())

        # Blank out range/lag combinations that are coincident with transmission
        range_lag_mask = np.zeros_like(visibilities, dtype=bool)
        # [num_pulses, num_ranges]
        for i, pulse_indices in enumerate(lags.tolist()):
            _, pulse_0_idx, _ = np.intersect1d(range_for_pulses[pulse_indices[0]], pulses_in_samples,
                                               return_indices=True)
            _, pulse_1_idx, _ = np.intersect1d(range_for_pulses[pulse_indices[1]], pulses_in_samples,
                                               return_indices=True)
            bad_ranges = np.union1d(pulse_0_idx, pulse_1_idx)
            range_lag_mask[..., i, bad_ranges] = True
        # Don't average over lag 0
        range_lag_mask[..., 0, :] = True
        range_lag_mask[..., -1, :] = True
        # Don't average over the zero baselines
        range_lag_mask[[i for i in range(num_antennas)], [i for i in range(num_antennas)]] = True
        # Calculate noise for each range and lag, by taking stddev over baselines of median over sequences
        # TODO: Median real and imag separately?
        noise = np.ma.std(np.ma.median(np.ma.array(visibilities, mask=range_lag_mask), axis=2), axis=(0, 1))

        # Average the baselines that are redundant
        # [num_unique_baselines, num_sequences, num_lags, num_ranges]
        unique_visibilities = np.zeros((len(self.unique_baselines.keys()), num_sequences, num_lags, num_ranges),
                                       dtype=np.complex64)
        for i, (_, repeat_indices) in enumerate(self.unique_baselines.items()):
            # repeat_indices is [i, j] indices for antenna_i, antenna_j that create baseline
            first_indices = [idx[0] for idx in repeat_indices]
            second_indices = [idx[1] for idx in repeat_indices]
            # This will grab [first_indices[0], second_indices[0]], then [first_indices[1], second_indices[1]], etc.
            # then calculate the median over the redundant baselines
            unique_visibilities[i] = (np.median(visibilities[first_indices, second_indices].real, axis=0) +
                                      1j * np.median(visibilities[first_indices, second_indices].imag, axis=0))

        # Average the visibilities across the pulse sequences
        # [num_unique_baselines, num_lags, num_ranges]
        avg_visibilities = np.median(unique_visibilities, axis=1)

        # Calculate SNR from zero lag, zero baseline power
        # [num_ranges]
        snr = 10 * np.log10(np.abs(avg_visibilities[0, 0, :]) / np.median(noise))

        # Apply Tikhonov Regularization
        # [num_ranges]
        alpha = 1 #/ np.power(10, snr / 10)      # 1 / snr, dimensions [num_ranges]

        # Calculate inverse singular values of forward matrix, with regularization
        # [num_ranges, num_visibilities]
        s_inv = self.s[np.newaxis, :] / (self.s[np.newaxis, :]**2 + alpha)#[:, np.newaxis]**2)

        # Calculate the reverse matrix
        # [num_ranges, num_brightnesses, num_visibilities]
        reverse_matrix = (self.v[np.newaxis, ...] * s_inv[:, np.newaxis, :]) @ self.uh[np.newaxis, ...]

        # Next, calculate the brightness estimate using the visibilities and the reverse matrix
        # v = Fb, so calculating here b = (F^-1) * v
        # v: [num_visibilities, num_lags, num_ranges]
        # F^-1: [num_ranges, num_brightnesses, num_visibilities]
        # brightness: [num_ranges, num_lags, num_brightnesses]
        brightness = np.einsum('rbv,vlr->rlb', reverse_matrix, avg_visibilities)

        record['data'] = brightness
        record['raw_visibilities'] = visibilities
        record['noise'] = noise
        record['s_inv'] = s_inv
        record['directions'] = self.angles
        record['data_dimensions'] = np.array(brightness.shape, dtype=np.int32)
        record['data_descriptors'] = np.bytes_(['num_ranges', 'num_lags', 'num_points'])

        return record
