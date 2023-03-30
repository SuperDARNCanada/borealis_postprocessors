# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains functions for adding a lag table and processing bfiq data into rawacf data.
"""
from collections import OrderedDict
from typing import Union

import numpy as np

from postprocessors import BaseConvert, ProcessBfiq2Rawacf


class ProcessIMPT(BaseConvert):
    """
    Class for conversion of Borealis antennas_iq files into rawacf files. This class inherits from
    BaseConvert, which handles all functionality generic to postprocessing borealis files.

    See Also
    --------
    ConvertFile
    BaseConvert
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
        super().__init__(infile, outfile, 'bfiq', 'rawacf', infile_structure, outfile_structure)
        self.averaging_method = averaging_method

        self.process_file()

    @staticmethod
    def process_record(record: OrderedDict, averaging_method: Union[None, str], **kwargs) -> OrderedDict:
        """
        Takes a record from an ImptTest bfiq file process into a rawacf record.

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
        STD_8P_LAG_TABLE = [[0, 0],
                            [42, 43],
                            [22, 24],
                            [24, 27],
                            [27, 31],
                            [22, 27],
                            [24, 31],
                            [14, 22],
                            [22, 31],
                            [14, 24],
                            [31, 42],
                            [31, 43],
                            [14, 27],
                            [0, 14],
                            [27, 42],
                            [27, 43],
                            [14, 31],
                            [24, 42],
                            [24, 43],
                            [22, 42],
                            [22, 43],
                            [0, 22],
                            [0, 24],
                            [43, 43]]
        record['lags'] = np.array(STD_8P_LAG_TABLE, dtype=np.uint32)
        record = ProcessBfiq2Rawacf.process_record(record, averaging_method)

        return record
