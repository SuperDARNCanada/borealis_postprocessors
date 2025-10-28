# Copyright 2021 SuperDARN Canada, University of Saskatchewan

"""
This file contains functions for downsampling widebeam experiments back to normalscan
"""
from collections import OrderedDict
from typing import Union
import numpy as np
from postprocessors import BaseConvert
import postprocessors
import os
import traceback
import h5py
from functools import partial
from multiprocessing import get_context
import pydarnio
import postprocessors.core.convert_base as cb
import postprocessors.core.restructure as rs
from postprocessors import conversion_exceptions
import datetime as dt

try:
    import cupy as xp
except ImportError:
    import numpy as xp
    cupy_available = False
else:
    cupy_available = True

import logging

postprocessing_logger = logging.getLogger('borealis_postprocessing')

class Widebeam2NormalScan(BaseConvert):
    """
    Class for conversion of Widebeam to normalscan for beam-broadening experiments. This class
    inherits from BaseConvert, which handles all functionality generic to postprocessing borealis files.
    #note array type input does not work

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

    def __init__(self, infile: str, outfile: str, infile_type: str, outfile_type: str, infile_structure: str, outfile_structure: str):
        """
        Initialize the attributes of the class.

        Parameters
        ----------
        infile: str
            Path to input file.
        outfile: str
            Path to output file.
        infile_type: str
            Type of data file. Types include:
            'antennas_iq'
            'bfiq'
            'rawacf'
        outfile_type: str
            Desired type of output data file. Same types as above.
        infile_structure: str
            Borealis structure of input file. Either 'array' or 'site'.
        outfile_structure: str
            Borealis structure of output file. Either 'array', 'site', or 'dmap'.
        """
        super().__init__(infile, outfile, infile_type, outfile_type, infile_structure, outfile_structure)
        self.process_file()


    @staticmethod
    def process_record(record: OrderedDict, **kwargs) -> OrderedDict:
        """
        Takes a record from a rawacf file processes from widebeam to replicate normalscan rawacf with one beam "active'
        per integration time
        1. Grabs the closest minute to first timestamp
        2. Finds how many seconds past the minute, in order to determine a beam number
        3. Based on the beam to keep, save that portion of record
        4. Return the updated record
        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing rawacf data and metadata

        Returns
        -------
        record: OrderedDict
            hdf5 record, downsampled to one beam
        """
        beam2keep = 0
        first_min = dt.datetime.fromtimestamp(record['sqn_timestamps'][0]).replace(second =0, microsecond = 0)
        timestamp = dt.datetime.fromtimestamp(record['sqn_timestamps'][0])
        diff = abs(first_min - timestamp).total_seconds()/record['int_time']
        beam2keep = int(round(diff))
        if beam2keep >= 16:
            beam2keep = 0

        # Separate the beam to keep
        record['beam_nums'] = np.array([np.uint32(beam2keep)])
        record['beam_azms'] = np.array([record['beam_azms'][beam2keep]])
        if "data_dimensions" in record:
            record['data_dimensions'][0] = 1
            dims = record['data_dimensions']
        else:
            record['correlation_dimensions'][0] = 1
            dims = record['correlation_dimensions']
        record['main_acfs'] = record['main_acfs'][beam2keep, :, :].reshape(dims)
        record['intf_acfs'] = record['intf_acfs'][beam2keep, :, :].reshape(dims)
        record['xcfs'] = record['xcfs'][beam2keep, :, :].reshape(dims)
        if beam2keep == 0:
            record['scan_start_marker'] = True
        else:
            record['scan_start_marker'] = False
        return record

    @staticmethod
    def dmap_to_dmap(file_to_process: str, processed_file: str, **kwargs) -> dict:
        """
        Uses the timestamp to determine which beam to keep for a given timestamp, to replicate normal scan
        1. For dmap input collects a list of timestamps
        2. Collects records that have the same timestamps into a new group
        3. For each timestamp, determine which beam should be saved
        4. Save the associated beam and remove the others
        5. Update the scan marker
        6. Save as dmap RAWACF

        Parameters
        ----------
        file_to_process: str
            File that should be processed.
        processed_file: str
            Output file name
        """

        sdarn_read = pydarnio.SDarnRead(file_to_process)
        data = sdarn_read.read_rawacf()

        grouped_records = []

        def get_timestamp(rec: dict) -> dt.datetime:
            """Builds a datetime object from the metadata of the DMAP record"""
            timestamp = dt.datetime(
                rec['time.yr'],
                rec['time.mo'],
                rec['time.dy'],
                rec['time.hr'],
                rec['time.mt'],
                rec['time.sc'],
                rec['time.us'],
                tzinfo=dt.timezone.utc,
            )
            return timestamp

        timestamps = set()
        for rec in data:  # Find the record names
            timestamps.add(get_timestamp(rec))
        timestamps = sorted(list(timestamps))
        num_beams = len(data) / len(timestamps)

        for tstamp in timestamps:  # group all records with identical timestamps
            concurrent_recs = []
            for rec in data:
                rec_time = get_timestamp(rec)
                if rec_time == tstamp:
                    concurrent_recs.append(rec)
            grouped_records.append(concurrent_recs)

        beam_to_keep = 0
        recs_kept = []
        for concurrent_recs in grouped_records:
            rec = concurrent_recs[beam_to_keep]
            rec['scan'] = np.int16(beam_to_keep == 0)
            beam_to_keep += 1
            if beam_to_keep >= num_beams:
                beam_to_keep = 0
            recs_kept.append(rec)
        pydarnio.SDarnWrite(recs_kept, processed_file).write_rawacf(processed_file)
