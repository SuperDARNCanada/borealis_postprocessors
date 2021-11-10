# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains functions for converting antennas_iq files
to rawacf files.
"""
from collections import OrderedDict
from typing import Union

from data_processing.convert_base import BaseConvert
from data_processing.antennas_iq_to_bfiq import ProcessAntennasIQ2Bfiq
from data_processing.bfiq_to_rawacf import ProcessBfiq2Rawacf


class ProcessAntennasIQ2Rawacf(BaseConvert):
    """
    Class for conversion of Borealis antennas_iq files. This includes both restructuring of
    data files, and processing into higher-level data files.

    See Also
    --------
    ConvertFile
    ConvertBfiq
    ConvertRawacf

    Attributes
    ----------
    filename: str
        The filename of the input antennas_iq file.
    output_file: str
        The file name of output file
    file_structure: str
        The write structure of the file. Structures include:
        'array'
        'site'
    final_structure: str
        The desired structure of the output file. Same structures as above.
    """

    def __init__(self, filename: str, output_file: str, file_structure: str, final_structure: str,
                 averaging_method: str = 'mean'):
        super().__init__(filename, output_file, 'antennas_iq', 'rawacf', file_structure, final_structure)
        self.averaging_method = averaging_method

        self.process_file()

    @staticmethod
    def process_record(record: OrderedDict, averaging_method: Union[None, str]) -> OrderedDict:
        """
        Takes a record from an antennas_iq file and beamforms the data.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing antennas_iq data and metadata
        averaging_method: Union[None, str]
            Method to use for averaging correlations across sequences. Acceptable methods are 'median' and 'mean'

        Returns
        -------
        record: OrderedDict
            hdf5 record, with new fields required by bfiq data format
        """
        record = ProcessAntennasIQ2Bfiq.process_record(record, averaging_method)
        record = ProcessBfiq2Rawacf.process_record(record, averaging_method)

        return record
