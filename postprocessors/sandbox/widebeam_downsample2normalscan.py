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
        self.infile = infile
        self.outfile = outfile
        self.infile_type = infile_type
        self.infile_structure = infile_structure
        self.outfile_type = outfile_type
        self.outfile_structure = outfile_structure



    @staticmethod
    def process_record(record: OrderedDict, **kwargs) -> OrderedDict:
        """
        Takes a record from a rawacf file process into a rawacf record.
        This method also keeps a single designated beam,

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing rawacf data and metadata
        beam_index: Dict
            A dictionary mapping timestamp to beam index

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
        else:
            record['correlation_dimensions'][0] = 1
        record['main_acfs'] = record['main_acfs'][beam2keep, :, :]
        record['intf_acfs'] = record['intf_acfs'][beam2keep, :, :]
        record['xcfs'] = record['xcfs'][beam2keep, :, :]
        if beam2keep == 0:
            record['scan_start_marker'] = True
        else:
            record['scan_start_marker'] = False
        return record

    @staticmethod
    def dmap_to_dmap(file_to_process: str, processed_file: str, **kwargs) -> dict:
        """
        Checks what beam index is associated to timestamp and keeps only that beam from a set of 16 records.

        Parameters
        ----------
        all_records: list
            list of timestamps
        record: dict
            dictionary containing rawacf data, where the key is timestamp and each entry is a list of 16 beams
            corresponding to one integration time/full FOV

        Returns
        -------
        beamedrec: list
            The downsampled record
        """

        sdarn_read = pydarnio.SDarnRead(file_to_process)
        data = sdarn_read.read_rawacf()

        record = dict()
        all_records = []  # record names

        for rec in data:  # Find the record names
            all_records.append(
                str(rec['time.yr']) + str(rec['time.mo']) + str(rec['time.dy']) + str(rec['time.hr']) + str(
                    rec['time.mt']) + str(rec['time.sc']) + str(rec['time.us']))

        all_records = list(dict.fromkeys(all_records))

        for i in all_records:  # reformat the records in 16 records per entry to better visualise FullFOV
            beam_rec = []
            for rec in data:
                rec_time = str(rec['time.yr']) + str(rec['time.mo']) + str(rec['time.dy']) + str(
                    rec['time.hr']) + str(rec['time.mt']) + str(rec['time.sc']) + str(rec['time.us'])
                if rec_time == i:
                    beam_rec.append(rec)
            record[i] = beam_rec

        cnt = 0  # initialize beam counter
        beamedrec = []
        for i in all_records:
            if cnt < 16:
                newrec = record[i][cnt]
                if cnt == 0: #flag the scan marker if at the start of scan
                    newrec['scan'] = np.int16(1)
                else:
                    newrec['scan'] = np.int16(0)
                cnt += 1
            else:
                cnt = 0
                newrec = record[i][cnt]
                if cnt == 0: #flag the scan marker if at the start of scan
                    newrec['scan'] = np.int16(1)
                else:
                    newrec['scan'] = np.int16(0)
                cnt += 1
            beamedrec.append(newrec) # add the beam to keep
        pydarnio.SDarnWrite(beamedrec, processed_file).write_rawacf(processed_file)
