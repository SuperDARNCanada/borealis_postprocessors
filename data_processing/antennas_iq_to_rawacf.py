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
    Class for conversion of Borealis antennas_iq files into rawacf files. This class inherits from
    BaseConvert, which handles all functionality generic to postprocessing borealis files.

    See Also
    --------
    ConvertFile
    BaseConvert
    ProcessAntennasIQ2Bfiq
    ProcessBfiq2Rawacf

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
        super().__init__(infile, outfile, 'antennas_iq', 'rawacf', infile_structure, outfile_structure)
        self.averaging_method = averaging_method

        self.process_file()

    @staticmethod
    def process_record(record: OrderedDict, averaging_method: Union[None, str]) -> OrderedDict:
        """
        Takes a record from an antennas_iq file process into a rawacf record.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing antennas_iq data and metadata
        averaging_method: Union[None, str]
            Method to use for averaging correlations across sequences. Acceptable methods are 'median' and 'mean'

        Returns
        -------
        record: OrderedDict
            hdf5 record, with new fields required by rawacf data format
        """
        record = ProcessAntennasIQ2Bfiq.process_record(record, averaging_method)
        record = ProcessBfiq2Rawacf.process_record(record, averaging_method)

        return record
