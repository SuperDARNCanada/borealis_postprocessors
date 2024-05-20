# Copyright 2024 SuperDARN Canada, University of Saskatchewan

"""
This file contains functions for frequency-domain analysis of high bandwidth antennas_iq files.
"""
from collections import OrderedDict

import numpy as np
import numpy.fft as fft
from postprocessors import BaseConvert


class FreqAnalysis(BaseConvert):
    """
    Class for conversion of Borealis antennas_iq data into downsampled frequency-domain data.

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
        The write structure of the file. Structures include:
        'array'
        'site'
    outfile_structure: str
        The desired structure of the output file. Same structures as above, plus 'dmap'.
    """

    def __init__(self, infile: str, outfile: str, infile_structure: str, outfile_structure: str, **kwargs):
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
        super().__init__(infile, outfile, "antennas_iq", "antennas_iq", infile_structure, outfile_structure)

    @classmethod
    def process_record(cls, record: OrderedDict, **kwargs) -> OrderedDict:
        data = record["data"]

        fs = 5e6 / 15  # Hz

        pulses = record["pulses"]
        tau = round(record["tau_spacing"] * 1e-6 * fs)  # puts into units of samples
        pulses_in_samples = [int(round(p * tau)) for p in pulses]
        pulse_dur = round(0.0006 * fs)
        mask = np.zeros(data.shape[-1], dtype=bool)
        for pulse in pulses_in_samples:
            start_mask = int(round(pulse - pulse_dur / 2))
            end_mask = int(round(pulse + pulse_dur / 2))
            if start_mask < 0:
                start_mask = 0
            if end_mask >= len(mask):
                end_mask = len(mask) - 1
            mask[start_mask: end_mask] = 1

        data[..., mask] = 0.0 + 0.0j
        masked_fft = np.sum(np.abs(fft.fftshift(fft.fft(data), axes=-1)), axis=0)

        freqs = fft.fftshift(fft.fftfreq(data.shape[-1], d=1/fs))
        df = freqs[1] - freqs[0]
        output_freq_resolution = 1000   # Hz
        sample_resolution = int(round(output_freq_resolution / df))

        kernel = np.ones((sample_resolution,)) / sample_resolution
        reduced_fft = np.array([np.convolve(masked_fft[i, :], kernel, mode='same')
                                for i in range(masked_fft.shape[0])])[..., ::sample_resolution]
        record["data"] = reduced_fft
        record["freqs"] = freqs[::sample_resolution] - df/2

        return record
