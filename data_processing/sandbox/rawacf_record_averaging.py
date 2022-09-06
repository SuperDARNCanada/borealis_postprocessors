# Copyright 2022 SuperDARN Canada, University of Saskatchewan

"""
This file contains functions for averaging multiple records of a rawacf file.
"""
from collections import OrderedDict
from typing import Union
import numpy as np

from data_processing.convert_base import BaseConvert


class AverageMultipleRawacfRecords(BaseConvert):
    """
    Class for averaging multiple rawacf records together. This class inherits from BaseConvert, which handles all
    functionality generic to postprocessing borealis files.

    See Also
    --------
    ConvertFile
    BaseConvert
    ProcessBfiq2Rawacf
    ProcessAntennasIQ2Rawacf

    Attributes
    ----------
    infile: str
        The filename of the input rawacf file.
    outfile: str
        The file name of output file
    infile_structure: str
        The write structure of the file. Structures include:
        'array'
        'site'
    outfile_structure: str
        The desired structure of the output file. Same structures as above, plus 'dmap'.
    """

    def __init__(self, infile: str, outfile: str, infile_structure: str, outfile_structure: str,
                 num_records: int = 2):
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
        num_records: int
            Number of records to average together. Default 2.
        """
        super().__init__(infile, outfile, 'rawacf', 'rawacf', infile_structure, outfile_structure)

        self._num_records = num_records

        self.process_file(avg_num=self._num_records)

    @staticmethod
    def process_record(record: OrderedDict, averaging_method: Union[None, str] = 'mean', **kwargs) -> OrderedDict:
        """
        Takes a record from a rawacf file and averages a specified number of adjacent records together.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing antennas_iq data and metadata
        averaging_method: Union[None, str]
            Method to use for averaging correlations across sequences. For this class, only 'mean' averaging is
            supported, as the median cannot be taken across multiple records after rawacf files have been made.
        kwargs:
            Supported key: 'extra_records'
            'extra_records' should be a list of OrderedDicts, which are the records to average.
        Returns
        -------
        record: OrderedDict
            hdf5 record
        """
        if 'extra_records' not in kwargs:
            print("No extra records given.")
            return record

        total_sequences = record['num_sequences']
        sqn_timestamps = list(record['sqn_timestamps'])
        int_time = record['int_time']
        noise_at_freq = list(record['noise_at_freq'])
        main_acfs = record['main_acfs'] * total_sequences
        intf_acfs = record['intf_acfs'] * total_sequences
        xcfs = record['xcfs'] * total_sequences

        for rec in kwargs['extra_records']:
            num_sequences = rec['num_sequences']
            total_sequences += num_sequences
            int_time += rec['int_time']
            noise_at_freq.extend(rec['noise_at_freq'])
            sqn_timestamps.extend(list(rec['sqn_timestamps']))
            main_acfs += rec['main_acfs'] * num_sequences
            intf_acfs += rec['intf_acfs'] * num_sequences
            xcfs += rec['xcfs'] * num_sequences

        main_acfs /= total_sequences
        intf_acfs /= total_sequences
        xcfs /= total_sequences

        record['main_acfs'] = np.array(main_acfs, dtype=np.complex64)
        record['intf_acfs'] = np.array(intf_acfs, dtype=np.complex64)
        record['xcfs'] = np.array(xcfs, dtype=np.complex64)
        record['int_time'] = int_time
        record['num_sequences'] = total_sequences
        record['sqn_timestamps'] = sqn_timestamps
        record['noise_at_freq'] = noise_at_freq

        return record
