# Copyright 2022 SuperDARN Canada, University of Saskatchewan

"""
This file contains functions for extracting sequence timestamps from any file type.
"""
from collections import OrderedDict
from typing import Union

from data_processing.convert_base import BaseConvert


class ExtractTimestamps(BaseConvert):
    """
    Class for extraction of sequence timestamps for bistatic experiments.
    This class inherits from BaseConvert, which handles all functionality generic to postprocessing borealis files.

    See Also
    --------
    ConvertFile
    BaseConvert
    ProcessAntennasIQ2Bfiq
    ProcessBfiq2Rawacf
    ProcessAntennasIQ2Rawacf

    Attributes
    ----------
    infile: str
        The filename of the input file.
    outfile: str
        The file name of output file
    infile_structure: str
        The write structure of the file. Structures include:
        'array'
        'site'
    outfile_structure: str
        The desired structure of the output file. Same structures as above, plus 'dmap'.
    """

    def __init__(self, infile: str, outfile: str, infile_type: str, outfile_type: str,
                 infile_structure: str, outfile_structure: str):
        """
        Initialize the attributes of the class.

        Parameters
        ----------
        infile: str
            Path to input file.
        outfile: str
            Path to output file.
        infile_type: str
            Borealis filetype of input file. 'antennas_iq', 'bfiq', or 'rawacf'.
        outfile_type: str
            Borealis filetype of output file. 'antennas_iq', 'bfiq', or 'rawacf'.
        infile_structure: str
            Borealis structure of input file. Either 'array' or 'site'.
        outfile_structure: str
            Borealis structure of output file. Either 'array', 'site', or 'dmap'.
        """
        super().__init__(infile, outfile, infile_type, outfile_type, infile_structure, outfile_structure)

        self.process_file()

    @staticmethod
    def process_record(record: OrderedDict, averaging_method: Union[None, str], **kwargs) -> OrderedDict:
        """
        Takes a record from a Borealis file and extract the sequence timestamps only.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing antennas_iq data and metadata
        averaging_method: Union[None, str]
            Method to use for averaging correlations across sequences. Acceptable methods are 'median' and 'mean'

        Returns
        -------
        record: OrderedDict
            hdf5 record, containing only the sequence timestamps.
        """
        new_record = OrderedDict()
        new_record['sqn_timestamps'] = record['sqn_timestamps']

        return new_record
