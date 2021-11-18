# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains functions for replacing cluttered lag0 ranges with
data from alternate lag0.
"""
from collections import OrderedDict
from typing import Union
import numpy as np

from data_processing.convert_base import BaseConvert


class ReprocessRawacfLag0(BaseConvert):
    """
    Class for reprocessing of Borealis rawacf hdf5 files to replace lag0 data from ranges which contain
    second-pulse clutter with alternate lag0 data.

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
        super().__init__(infile, outfile, 'rawacf', 'rawacf', infile_structure, outfile_structure)

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
        # Get the correlation data
        main_acfs = record['main_acfs']
        intf_acfs = record['intf_acfs']
        xcfs = record['xcfs']

        correlation_descriptors = record['correlation_descriptors']
        correlation_dimensions = record['correlation_dimensions']
        
        main_acfs = main_acfs.reshape(correlation_dimensions)
        intf_acfs = intf_acfs.reshape(correlation_dimensions)
        xcfs = xcfs.reshape(correlation_dimensions)
        
        # Convert tau spacing into units of samples
        tau_in_samples = record['tau_spacing'] * 1e-6 * record['rx_sample_rate']

        # First range offset in units of samples
        sample_off = record['first_range_rtt'] * 1e-6 * record['rx_sample_rate']
        sample_off = xp.int32(sample_off)

        # Start of second pulse in units of samples
        second_pulse_sample_num = np.int32(tau_in_samples) * record['pulses'][1] - sample_off - 1

        # Replace all ranges which are contaminated by the second pulse for lag 0
        # with the data from those ranges after the final pulse.
        main_acfs[:, second_pulse_sample_num:, 0] = main_acfs[:, second_pulse_sample_num:, -1]
        intf_acfs[:, second_pulse_sample_num:, 0] = intf_acfs[:, second_pulse_sample_num:, -1]
        xcfs[:, second_pulse_sample_num:, 0] = xcfs[:, second_pulse_sample_num:, -1]
        
        record['main_acfs'] = main_acfs
        record['intf_acfs'] = intf_acfs
        record['xcfs'] = xcfs

        return record
