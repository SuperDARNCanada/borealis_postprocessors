# Copyright 2024 SuperDARN Canada, University of Saskatchewan

"""
This file contains functions for imaging antennas_iq files to rawacf files,
using the Spherical Wave Harmonic Transform (Carozzi, 2015) to solve the imaging inverse problem.
"""
import copy
import os
from collections import OrderedDict
import numpy as np
import h5py
import healpy as hp

from postprocessors import AntennasIQ2Rawacf, AntennasIQ2Bfiq


class SWHTImaging(AntennasIQ2Rawacf):
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

    def __init__(self, infile: str, outfile: str, coeffs_file: str):
        """
        Initialize the attributes of the class.

        Parameters
        ----------
        infile: str
            Path to input file.
        outfile: str
            Path to output file.
        coeffs_file: str
            Path to file with coefficient matrices.
        """
        super().__init__(infile, outfile, 'site', 'site')

        self.freq_hz, self.unique_baselines = self.setup_inversion(infile)

        if not os.path.isfile(coeffs_file):
            # Default to a healpix grid, upper hemisphere only
            nside = 64
            lmax = 85
            npix = hp.nside2npix(nside)
            theta, phi = hp.pix2ang(nside=nside, ipix=np.arange(npix // 2))     # in radians
            fov = np.stack([phi, theta], axis=1)
            baselines = np.array(sorted(self.unique_baselines.keys()), dtype=np.float32) * self.freq_hz / 299792458
            generate_coeffs(coeffs_file, self.freq_hz, fov, lmax, baselines=baselines)

        f = h5py.File(coeffs_file, 'r')
        coeffs_group = f['coeffs']
        coeffs = coeffs_group['85'][()]
        f.close()

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
            data_dimensions = record['data_dimensions'][:]

            # [x, y] offsets for each antenna
            antenna_locations = np.zeros((data_dimensions[0], 3), dtype=np.float32)
            antenna_locations[:16, 0] = np.arange(16) * 15.24  # main array, leftmost is at [0, 0]
            antenna_locations[16:, 0] = np.arange(6, 10) * 15.24  # intf array
            antenna_locations[16:, 1] = -100.0  # intf array 100m behind main array

            # [num_antennas, num_antennas, 3] for [u, v, w] from antenna_i - antenna_j
            baselines = antenna_locations[:, np.newaxis, :] - antenna_locations[np.newaxis, :, :]

            # Group all redundant baselines, where (x, y) are keys and indices in baselines are the values
            unique_baselines = OrderedDict()
            for i in range(baselines.shape[0]):
                for j in range(baselines.shape[1]):
                    coords = baselines[i, j].round(decimals=3)
                    if (coords[0], coords[1], coords[2]) in unique_baselines.keys():
                        unique_baselines[(coords[0], coords[1], coords[2])].append((i, j))
                    else:
                        unique_baselines[(coords[0], coords[1], coords[2])] = [(i, j)]
            # Result is a dictionary where keys are unique baselines, and values are lists of indices that are redundant

            # Convert from distance to phase difference
            freq = record.attrs['freq'] * 1000  # freq is stored in kHz

            return freq, unique_baselines

    def process_record(self, record: OrderedDict, **kwargs) -> OrderedDict:
        """
        See AntennasIQ2Rawacf for description.
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
            visibilities[..., i, :] = np.einsum('asr,bsr->absr',  # antenna sequence range x antenna sequence range
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
        joint_baselines_and_indices = []
        for coord, repeat_idx in self.unique_baselines.items():
            joint_baselines_and_indices.append([coord, repeat_idx])     # Sort by the coordinate tuple
        for i, joint_obj in enumerate(sorted(joint_baselines_and_indices, key=lambda x: x[0])):
            repeat_indices = joint_obj[-1]  # Tuple of [coords_tuple, repeat_indices]
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

        # Next, calculate the brightness estimate using the visibilities and the coeffs matrix
        brightness = np.einsum('rbv,vlr->rlb', coeffs, avg_visibilities)

        record['data'] = brightness
        record['raw_visibilities'] = visibilities
        record['noise'] = noise
        record['directions'] = fov
        record['data_dimensions'] = np.array(brightness.shape, dtype=np.int32)
        record['data_descriptors'] = np.bytes_(['num_ranges', 'num_lags', 'num_points'])

        return record


def uvw_to_rtp(u, v, w):
    xy = u*u + v*v
    r = np.sqrt(xy + w*w)
    theta = np.arccos(w / r)
    phi = np.arctan2(v, u)

    return r, theta, phi


def coords_to_baselines(x, y, z, wavelength):
    coords = np.stack([x, y, z], axis=1)
    diffs = (coords[np.newaxis, ...] - coords[:, np.newaxis, :]) / wavelength
    idx = np.triu_indices(diffs.shape[0], 1)

    return diffs[idx][0], diffs[idx][1], diffs[idx][2]


def generate_coeffs(filename, freq_hz, fov, lmax=85, **kwargs):
    """
    Generates coefficient matrix for fast SWHT and stores in filename.

    Parameters
    ----------
    filename: str
        Path to file for storing coefficient matrices
    freq_hz: float
        Rx frequency in Hz
    fov: np.ndarray
        Array of [num_points, 2] defining the field of view, where the last dimension is azimuth, colatitude coordinates
    lmax: int
        Maximum spherical harmonic degree to calculate
    **kwargs: dict
        ant_coords: np.ndarray
            Array of antenna coordinates, as 2D array of [num_antennas, 3] where last dimension is x, y, z coordinates
        baselines: np.ndarray
            Array of visibility baselines, as 2D array of [num_baselines, 3] where last dimension is u, v, w coordinates
    """
    wavelength = 299792458 / freq_hz
    ko = 2 * np.pi / wavelength

    if 'ant_coords' not in kwargs and 'baselines' not in kwargs:
        raise RuntimeError("Must specify one of `ant_coords` or `baselines`")
    elif 'ant_coords' in kwargs and 'baselines' in kwargs:
        raise RuntimeError("Must specify only one of `ant_coords` or `baselines`")
    elif 'ant_coords' in kwargs:
        ant_coords = kwargs.get('ant_coords')
        u, v, w = coords_to_baselines(ant_coords[0, :],
                                      ant_coords[1, :],
                                      ant_coords[2, :],
                                      wavelength)
    else:
        baselines = kwargs.get('baselines')
        u, v, w = baselines[:, 0], baselines[:, 1], baselines[:, 2]

    r, t, p = uvw_to_rtp(u, v, w)
    r *= wavelength  # Since r, t, p was converted from u, v, w, we need the *wavelength back to match SWHT algorithm
    create_coeffs(filename, fov, lmax, wavelength, np.array([u, v, w]))
    calculate_coeffs(filename, fov, ko, r, t, p, lmax)


def create_coeffs(filename, fov, lmax, wavelength, baselines):
    """
    Creates a new HDF5 file for storing coefficient matrices.

    Parameters
    ----------
    filename: str
        Path to file
    baselines: np.ndarray
        Array of visibility baselines, as 2D array of [num_baselines, 3] where last dimension is u, v, w coordinates
    fov: np.ndarray
        Array of [num_points, 2] defining the field of view, where the last dimension is azimuth, colatitude coordinates
    lmax: int
        Maximum spherical harmonic degree to calculate
    wavelength: float
        Radio wavelength in meters.
    """
    f = h5py.File(filename, 'w')
    f.create_dataset('fov', data=fov)
    f.create_dataset('lmax', data=lmax)
    f.create_dataset('wavelength', data=wavelength)
    f.create_dataset('baselines', data=baselines)
    f.close()


def calculate_coeffs(filename, fov, ko, r, theta, phi, lmax):
    """
    Calculates coefficient matrices for SWHT.

    Parameters
    ----------
    filename: str
        Path to file that coefficients are saved in.
    fov: np.ndarray
        Array of [num_points, 2] defining the field of view, where the last dimension is azimuth, colatitude coordinates
    ko: float
        Wavenumber (2 * pi / wavelength) of radio wave
    r: np.ndarray
        Radial component of baselines, in meters
    theta: np.ndarray
        Colatitude component of baselines, in radians
    phi: np.ndarray
        Azimuthal component of baselines, in radians
    lmax: int
        Maximum spherical harmonic degree to calculate. Max 85

    Returns
    -------

    """
    from scipy.special import sph_harm, spherical_jn
    coeffs = np.zeros((fov.shape[0], len(r)), dtype=np.complex128)

    if lmax > 85:
        raise RuntimeError("Harmonic degree too large, solution would be numerically unstable")
    for l in range(lmax + 1):
        constant = ko * ko / (2 * np.pi * np.pi * ((-1j) ** (l % 4)))
        for m in range(-l, l + 1):
            # constant * Y_{l,m}(fov) * j_l(ko*r) * Y^*_{l,m}(theta, phi)
            coeffs += constant * \
                      np.repeat(sph_harm(m, l, fov[1], fov[0])[:, np.newaxis], len(r), axis=1) * \
                      np.repeat(spherical_jn(l, ko * r) *
                                np.conjugate(sph_harm(m, l, phi, theta))[np.newaxis, :],
                                fov.shape[0], axis=0)
        if l in range(5, 86, 10):
            append_coeffs(filename, l, coeffs)


def append_coeffs(filename, degree, coeffs):
    f = h5py.File(filename, 'a')
    f.create_dataset(f'coeffs/{degree:02d}', data=coeffs)
    f.close()
