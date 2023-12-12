# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains functions for converting antennas_iq files
to rawacf files.
"""
from collections import OrderedDict

from postprocessors import BaseConvert, AntennasIQ2Bfiq, Bfiq2Rawacf


class AntennasIQ2Rawacf(AntennasIQ2Bfiq, Bfiq2Rawacf):
    """
    Class for conversion of Borealis antennas_iq files into rawacf files. This class inherits from
    BaseConvert, which handles all functionality generic to postprocessing borealis files.

    See Also
    --------
    ConvertFile
    BaseConvert
    AntennasIQ2Bfiq
    Bfiq2Rawacf

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
        The desired structure of the output file. Same structures as above, plus 'dmap'.
    """

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
            Borealis structure of output file. Either 'array', 'site', or 'dmap'.
        """
        BaseConvert.__init__(self, infile, outfile, 'antennas_iq', 'rawacf', infile_structure, outfile_structure)

    @classmethod
    def process_record(cls, record: OrderedDict, **kwargs) -> OrderedDict:
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
        record = super(AntennasIQ2Rawacf, cls).process_record(record, **kwargs) # Calls AntennasIQ2Bfiq.process_record()
        record = super(AntennasIQ2Bfiq, cls).process_record(record, **kwargs)   # Calls Bfiq2Rawacf.process_record()

        return record
