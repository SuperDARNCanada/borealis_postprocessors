# Copyright 2021 SuperDARN Canada, University of Saskatchewan

"""
This file contains functions for converting antennas_iq files from widebeam experiments
to bfiq files.
"""
from collections import OrderedDict
from typing import Union
import numpy as np

from postprocessors import BaseConvert, AntennasIQ2Bfiq


class WidebeamAntennasIQ2Bfiq(BaseConvert):
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
    beam_azms: list[float]
        List of all beam directions (in degrees) to reprocess into
    beam_nums: list[uint]
        List describing beam order. Numbers in this list correspond to indices of beam_azms
    """

    def __init__(self, infile: str, outfile: str, infile_structure: str, outfile_structure: str,
                 beam_azms: Union[list, None] = None, beam_nums: Union[list, None] = None):
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
        beam_azms: list[float]
            List of all beam directions (in degrees) to reprocess into
        beam_nums: list[uint]
            List describing beam order. Numbers in this list correspond to indices of beam_azms
        """
        super().__init__(infile, outfile, 'antennas_iq', 'bfiq', infile_structure, outfile_structure)

        # Use default 16-beam arrangement if no beams are specified
        if beam_azms is None:
            # Add extra phases here
            # STD_16_BEAM_ANGLE from superdarn_common_fields
            self.beam_azms = [-24.3, -21.06, -17.82, -14.58, -11.34, -8.1, -4.86, -1.62, 1.62, 4.86, 8.1, 11.34, 14.58,
                              17.82, 21.06, 24.3]

            # STD_16_FORWARD_BEAM_ORDER
            self.beam_nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        else:
            self.beam_azms = beam_azms
            # Step through beam_azms sequentially if not specified
            if beam_nums is None:
                self.beam_nums = range(len(beam_azms))
            elif len(beam_nums) != len(beam_azms):
                raise ValueError('beam_nums should be same length as beam_azms.')
            else:
                self.beam_nums = beam_nums

        self.process_file(beam_azms=self.beam_azms, beam_nums=self.beam_nums)

    @staticmethod
    def process_record(record: OrderedDict, averaging_method: Union[None, str], **kwargs) -> OrderedDict:
        """
        Takes a record from an antennas_iq file process into a bfiq record.
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
        record['first_range'] = AntennasIQ2Bfiq.calculate_first_range(record)
        record['first_range_rtt'] = AntennasIQ2Bfiq.calculate_first_range_rtt(record)
        record['lags'] = AntennasIQ2Bfiq.create_lag_table(record)
        record['range_sep'] = AntennasIQ2Bfiq.calculate_range_separation(record)
        record['num_ranges'] = AntennasIQ2Bfiq.get_number_of_ranges(record)

        record['beam_azms'] = np.float64(kwargs['beam_azms'])
        record['beam_nums'] = np.uint32(kwargs['beam_nums'])

        record['data'] = AntennasIQ2Bfiq.beamform_data(record)
        record['data_descriptors'] = AntennasIQ2Bfiq.get_data_descriptors()
        record['data_dimensions'] = AntennasIQ2Bfiq.get_data_dimensions(record)
        record['antenna_arrays_order'] = AntennasIQ2Bfiq.change_antenna_arrays_order()

        return record
