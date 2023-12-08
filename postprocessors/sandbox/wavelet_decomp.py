# Copyright 2021 SuperDARN Canada, University of Saskatchewan

"""
This file contains functions for converting antennas_iq files from widebeam experiments
to bfiq files.
"""
from collections import OrderedDict
from typing import Union
import numpy as np
import pywt
import dtcwt

from postprocessors import BaseConvert


class WaveletDecomposition(BaseConvert):
    """
    Class for wavelet decomposition of Borealis antennas_iq files.
    This class inherits from BaseConvert, which handles all functionality generic to postprocessing borealis files.

    See Also
    --------
    ConvertFile
    BaseConvert
    ProcessAntennasIQ2Bfiq

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
    """

    def __init__(self, infile: str, outfile: str, infile_structure: str):
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
        """
        super().__init__(infile, outfile, 'antennas_iq', 'bfiq', infile_structure, 'site')

        self.process_file()

    @staticmethod
    def process_record(record: OrderedDict, averaging_method: Union[None, str], **kwargs) -> OrderedDict:
        """
        Takes a record from an antennas_iq file and decomposes the data from each antenna individually.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing antennas_iq data and metadata
        averaging_method: Union[None, str]
            Method to use for averaging correlations across sequences. Acceptable methods are 'median' and 'mean'

        Returns
        -------
        record: OrderedDict
            hdf5 record
        """
        # record['first_range'] = ProcessAntennasIQ2Bfiq.calculate_first_range(record)
        # record['first_range_rtt'] = ProcessAntennasIQ2Bfiq.calculate_first_range_rtt(record)
        # record['lags'] = ProcessAntennasIQ2Bfiq.create_lag_table(record)
        # record['range_sep'] = ProcessAntennasIQ2Bfiq.calculate_range_separation(record)
        # record['num_ranges'] = ProcessAntennasIQ2Bfiq.get_number_of_ranges(record)
        #
        # record['beam_azms'] = np.float64(kwargs['beam_azms'])
        # record['beam_nums'] = np.uint32(kwargs['beam_nums'])
        #
        # record['data'] = ProcessAntennasIQ2Bfiq.beamform_data(record)
        # record['data_descriptors'] = ProcessAntennasIQ2Bfiq.get_data_descriptors()
        # record['data_dimensions'] = ProcessAntennasIQ2Bfiq.get_data_dimensions(record)
        # record['antenna_arrays_order'] = ProcessAntennasIQ2Bfiq.change_antenna_arrays_order()

        num_antennas, num_sequences, num_samps = record['data_dimensions']
        signal = np.reshape(record['data'], record['data_dimensions'])

        wavelet = pywt.Wavelet('db4')
        # print(wavelet)
        detail1_len = int(np.floor((num_samps + wavelet.dec_len - 1) / 2))
        detail2_len = int(np.floor((detail1_len + wavelet.dec_len - 1) / 2))
        detail3_len = int(np.floor((detail2_len + wavelet.dec_len - 1) / 2))
        detail4_len = int(np.floor((detail3_len + wavelet.dec_len - 1) / 2))
        approx_len = int(np.floor((detail3_len + wavelet.dec_len - 1) / 2))

        approx = np.empty((num_antennas, num_sequences, approx_len), dtype=np.complex64)
        detail1 = np.empty((num_antennas, num_sequences, detail1_len), dtype=np.complex64)
        detail2 = np.empty((num_antennas, num_sequences, detail2_len), dtype=np.complex64)
        detail3 = np.empty((num_antennas, num_sequences, detail3_len), dtype=np.complex64)
        detail4 = np.empty((num_antennas, num_sequences, detail4_len), dtype=np.complex64)

        for i in range(num_antennas):
            for j in range(num_sequences):
                approx[i, j], detail4[i, j], detail3[i, j], detail2[i, j], detail1[i, j] = \
                    pywt.wavedec(signal[i, j], 'db4', level=4)
        record['approx'] = approx
        record['detail4'] = detail4
        record['detail3'] = detail3
        record['detail2'] = detail2
        record['detail1'] = detail1

        # num_antennas, num_sequences, num_samps = record['data_dimensions']
        # signal = np.reshape(record['data'], record['data_dimensions'])
        #
        # transform = dtcwt.Transform1d()
        #
        # vecs_t = []
        # for i in range(num_antennas):
        #     for j in range(num_sequences):
        #         vecs = np.array([signal[i, j].real, signal[i, j].imag], dtype=np.float32)
        #         vecs_t.append(transform.forward(vecs, nlevels=4))
        #         # print(vecs_t[-1].lowpass.shape, vecs_t[-1].lowpass.dtype)
        #         # print(vecs_t[-1].highpasses[-1].shape, vecs_t[-1].highpasses[-1].dtype)
        #
        # print(vecs_t[0].lowpass.shape)
        # approx = np.empty((num_antennas, num_sequences, vecs_t[0].lowpass.shape[-1]), dtype=np.complex64)
        # print('approx shape ', approx.shape)
        # detail1 = np.empty((num_antennas, num_sequences, vecs_t[0].highpasses[0].shape[-1]), dtype=np.complex64)
        # detail2 = np.empty((num_antennas, num_sequences, vecs_t[0].highpasses[1].shape[-1]), dtype=np.complex64)
        # detail3 = np.empty((num_antennas, num_sequences, vecs_t[0].highpasses[2].shape[-1]), dtype=np.complex64)
        # detail4 = np.empty((num_antennas, num_sequences, vecs_t[0].highpasses[3].shape[-1]), dtype=np.complex64)
        #
        # for i, vec in enumerate(vecs_t):
        #     ant_idx = i // num_sequences
        #     seq_idx = i % num_sequences
        #     print('ant_idx ', ant_idx, 'seq_idx ', seq_idx)
        #     approx[ant_idx, seq_idx] = vec.lowpass[0] + 1j * vec.lowpass[1]
        #     detail1[ant_idx, seq_idx] = vec.highpasses[0][0]
        #     detail2[ant_idx, seq_idx] = vec.highpasses[1][0]
        #     detail3[ant_idx, seq_idx] = vec.highpasses[2][0]
        #     detail4[ant_idx, seq_idx] = vec.highpasses[3][0]
        #
        # record['approx'] = approx
        # record['detail4'] = detail4
        # record['detail3'] = detail3
        # record['detail2'] = detail2
        # record['detail1'] = detail1

        return record
