# Copyright 2021 SuperDARN Canada, University of Saskatchewan

"""
This file contains functions for converting antennas_iq files from widebeam experiments
to rawacf files.
"""
from collections import OrderedDict
from typing import Union

from data_processing.convert_base import BaseConvert
from data_processing.antennas_iq_to_bfiq import ProcessAntennasIQ2Bfiq
from data_processing.bfiq_to_rawacf import ProcessBfiq2Rawacf


class ProcessWidebeamAntennasIQ2Rawacf(BaseConvert):
    """
    Class for conversion of Borealis antennas_iq files into rawacf files for beam-broadening experiments. This class
    inherits from BaseConvert, which handles all functionality generic to postprocessing borealis files.

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
        This method also beamforms the antennas_iq data into each of the 16 standard SuperDARN beams,
        regardless of the beams transmitted, overwriting the fields 'beam_azms' and 'beam_nums'.

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
        record['first_range'] = ProcessAntennasIQ2Bfiq.calculate_first_range(record)
        record['first_range_rtt'] = ProcessAntennasIQ2Bfiq.calculate_first_range_rtt(record)
        record['lags'] = ProcessAntennasIQ2Bfiq.create_lag_table(record)
        record['range_sep'] = ProcessAntennasIQ2Bfiq.calculate_range_separation(record)
        record['num_ranges'] = ProcessAntennasIQ2Bfiq.get_number_of_ranges(record)

        # Add extra phases here
        # STD_16_BEAM_ANGLE from superdarn_common_fields
        beam_azms = [-26.25, -22.75, -19.25, -15.75, -12.25, -8.75, -5.25, -1.75,
                     1.75, 5.25, 8.75, 12.25, 15.75, 19.25, 22.75, 26.25]

        # STD_16_FORWARD_BEAM_ORDER
        beam_nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        record['beam_azms'] = beam_azms
        record['beam_nums'] = beam_nums

        record['data'] = ProcessAntennasIQ2Bfiq.beamform_data(record)
        record['data_descriptors'] = ProcessAntennasIQ2Bfiq.get_data_descriptors()
        record['data_dimensions'] = ProcessAntennasIQ2Bfiq.get_data_dimensions(record)
        record['antenna_arrays_order'] = ProcessAntennasIQ2Bfiq.change_antenna_arrays_order()

        record = ProcessBfiq2Rawacf.process_record(record, averaging_method)

        return record
