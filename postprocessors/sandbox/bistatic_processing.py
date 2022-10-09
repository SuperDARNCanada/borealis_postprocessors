# Copyright 2022 SuperDARN Canada, University of Saskatchewan

"""
This file contains functions for processing bistatic_test antennas_iq files to rawacf.
"""
from collections import OrderedDict
from typing import Union
import numpy as np
import deepdish as dd

from postprocessors import BaseConvert, ProcessAntennasIQ2Rawacf


class BistaticProcessing(BaseConvert):
    """
    Class for processing bistatic_test antennas_iq files to rawacf. To properly use this class, you first must
    extract the timestamps from the transmitting radar using the ExtractTimestamps class. These are cross-referenced
    against the timestamps in the receiving radar file (infile for this class), so that only sequences which occurred
    at the same time for both radars are processed.

    This class inherits from BaseConvert, which handles all functionality generic to postprocessing borealis files.

    See Also
    --------
    ConvertFile
    BaseConvert
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

    def __init__(self, infile: str, outfile: str, infile_structure: str, outfile_structure: str, timestamps_file: str):
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
        timestamps_file: str
            Path to file containing transmitting radar's sequence timestamps
        """
        super().__init__(infile, outfile, 'antennas_iq', 'rawacf', infile_structure, outfile_structure)

        timestamps_opened = dd.io.load(timestamps_file)   # Load the whole file in
        keys = sorted(list(timestamps_opened.keys()))

        timestamps = np.concatenate([timestamps_opened[k]['sqn_timestamps'] for k in keys])
        timestamps = np.around(timestamps, decimals=6)      # round to nearest microsecond

        self.process_file(timestamps=timestamps)

    @staticmethod
    def process_record(record: OrderedDict, averaging_method: Union[None, str], **kwargs) -> OrderedDict:
        """
        Takes a record from a Borealis file and only processes the sequences which were transmitted by
        the transmitting radar site.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing antennas_iq data and metadata
        averaging_method: Union[None, str]
            Method to use for averaging correlations across sequences. Acceptable methods are 'median' and 'mean'
        **kwargs: dict
            timestamps: ndarray
                1D array of timestamps in seconds past epoch

        Returns
        -------
        record: OrderedDict
            hdf5 record
        """
        tx_timestamps = kwargs['timestamps']

        rx_tstamps = np.around(record['sqn_timestamps'], decimals=6)   # Round to nearest microsecond

        # Get the tx timestamps that occur during this record
        tx_indices = list(np.argwhere(rx_tstamps[0] <= tx_timestamps <= rx_tstamps[-1]))
        tx_times = set(tx_timestamps[tx_indices])

        # Keep all indices from rx_tstamps that have a corresponding tx timestamp
        keep_indices = []
        for t in range(len(rx_tstamps)):
            if rx_tstamps[t] in tx_times:
                keep_indices.append(t)

        # Extract the good data
        data_dimensions = record['data_dimensions']
        data = record['data'].reshape(data_dimensions)
        record['data'] = data[:, keep_indices, :].flatten()   # [num_antennas, num_sequences, num_samps]

        # Update the metadata
        data_dimensions[1] = np.uint32(len(keep_indices))
        record['data_dimensions'] = data_dimensions
        record['num_sequences'] = np.int64(len(keep_indices))
        record['sqn_timestamps'] = record['sqn_timestamps'][keep_indices]

        record = ProcessAntennasIQ2Rawacf.process_record(record, averaging_method)

        return record
