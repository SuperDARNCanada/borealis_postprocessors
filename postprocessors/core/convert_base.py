# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains functions for converting antennas_iq files
to bfiq files.
"""
import os
import subprocess as sp
from collections import OrderedDict
from typing import Union
import deepdish as dd
import h5py
from functools import partial
from multiprocessing import get_context

from postprocessors.core.restructure import restructure, convert_to_numpy, FILE_STRUCTURE_MAPPING
from postprocessors import conversion_exceptions

try:
    import cupy as xp
except ImportError:
    import numpy as xp
    cupy_available = False
else:
    cupy_available = True

import logging

postprocessing_logger = logging.getLogger('borealis_postprocessing')


def processing_machine(idx: int, filename: str, record_keys: list, records_per_process: int, averaging_method: str,
                       processing_fn, **kwargs):
    """
    Helper function for processing a single record. It is defined here to facilitate multiprocessing.

    Parameters
    ----------
    idx: int
        Index into record_keys which tells processing_machine() which record to process
    filename: str
        Name of the HDF5 file with records to process.
    record_keys: list
        List of all top-level keys of the HDF5 file.
    records_per_process: int
        Number of records to process per call to this function.
    averaging_method: str
        Method for averaging rawacf data. Either 'mean' or 'median'
    processing_fn: callable
        Function to call to process a record.
    kwargs: dict
        Key-word arguments to pass to processing_fn

    Returns
    -------
    formatted_record, idx: properly-formatted processed record and the index which was processed.
    """
    record_dict = dd.io.load(filename, f'/{record_keys[idx]}')
    record_list = []  # List of all 'extra' records to process

    # If processing multiple records at a time, get all the records ready
    if records_per_process > 1:
        for num in range(idx + 1, min(idx + records_per_process, len(record_keys))):
            record_list.append(dd.io.load(filename, f'/{record_keys[num]}'))

    processed_record = processing_fn(record_dict, averaging_method, extra_records=record_list, **kwargs)

    if processed_record is None:
        return None, idx
    else:
        # Convert to numpy arrays for saving to file with deepdish
        formatted_record = convert_to_numpy(processed_record)
        return formatted_record, idx


class BaseConvert(object):
    """
    Class for converting Borealis filetypes of all structures. This class abstracts and redirects
    the file being converted to the correct class (ProcessAntennasIQ2Bfiq, ProcessAntennasIQ2Rawacf,
    or ProcessBfiq2Rawacf).

    See Also
    --------
    ProcessAntennasIQ2Bfiq
    ProcessAntennasIQ2Rawacf
    ProcessBfiq2Rawacf
    ConvertFile

    Attributes
    ----------
    infile: str
        The filename of the input file containing SuperDARN data.
    outfile: str
        The file name of output file
    infile_type: str
        Type of data file. Types include:
        'antennas_iq'
        'bfiq'
        'rawacf'
    outfile_type: str
        Desired type of output data file. Same types as above.
    infile_structure: str
        The write structure of the file. Structures include:
        'array'
        'site'
        'iqdat' (bfiq only)
        'dmap' (rawacf only)
        All borealis files are either 'site' or 'array' structured.
    outfile_structure: str
        The desired structure of the output file. Same structures as
        above.
    """
    def __init__(self, infile: str, outfile: str, infile_type: str, outfile_type: str, infile_structure: str,
                 outfile_structure: str):
        """
        Initializes the attributes of the class.

        Parameters
        ----------
        infile: str
            Path to the input file
        outfile: str
            Path to the output file
        infile_type: str
            Borealis file type of input file. Supported types are:
            'antennas_iq'
            'bfiq'
            'rawacf'
        outfile_type: str
            Borealis file type of output file. Supported types are same as for file_type.
        infile_structure: str
            Borealis file structure of input file. Supported structures are:
            'array'
            'site'
        outfile_structure: str
            Borealis file structure of output file. Supported structures are:
            'array'
            'site'
            'iqdat' (bfiq only)
            'dmap' (rawacf only)
        """
        self.infile = infile
        self.outfile = outfile
        self.infile_type = infile_type
        self.infile_structure = infile_structure
        self.outfile_type = outfile_type
        self.outfile_structure = outfile_structure
        self.check_args()

        self.averaging_method = None
        self._temp_files = []

    def check_args(self):

        if self.infile_structure not in FILE_STRUCTURE_MAPPING[self.infile_type]:
            raise conversion_exceptions.ImproperFileStructureError(
                f'Input file structure "{self.infile_structure}" is not compatible with input file type '
                f'"{self.infile_type}": Valid structures for {self.infile_type} are '
                f'{FILE_STRUCTURE_MAPPING[self.infile_type]}'
            )
        if self.outfile_structure not in FILE_STRUCTURE_MAPPING[self.outfile_type]:
            raise conversion_exceptions.ImproperFileStructureError(
                f'Output file structure "{self.outfile_structure}" is not compatible with output file type '
                f'"{self.outfile_type}": Valid structures for {self.outfile_type} are '
                f'{FILE_STRUCTURE_MAPPING[self.outfile_type]}'
            )
        if self.infile_structure not in ['array', 'site']:
            raise conversion_exceptions.ConversionUpstreamError(
                f'Input file structure "{self.infile_structure}" cannot be reprocessed into any other format.'
            )

    def process_file(self, **kwargs):
        """
        Applies appropriate downstream processing to convert between file types (for site-structured
        files only). The processing chain is as follows:
        1. Restructure to site format
        2. Apply appropriate downstream processing
        3. Restructure to final format
        4. Remove all intermediate files created along the way

        See Also
        --------
        ProcessAntennasIQ2Bfiq
        ProcessAntennasIQ2Rawacf
        ProcessBfiq2Rawacf
        """

        if os.path.isfile(self.outfile) and not kwargs.get('force', False):
            choice = input(f'Output file {self.outfile} already exists. Proceed anyway? Only records which don\'t '
                           f'exist in output file will be processed. (y/n): ')
            if choice[0] not in ['y', 'Y']:
                return 0

        try:
            # Restructure to 'site' format if necessary
            if self.infile_structure != 'site':
                file_to_process = f'{self.infile}.site.tmp'
                self._temp_files.append(file_to_process)
                # Restructure file to site format for processing
                postprocessing_logger.info(f'Restructuring file {self.infile} --> {file_to_process}')
                restructure(self.infile, file_to_process, self.infile_type, self.infile_structure, 'site')
            else:
                file_to_process = self.infile

            # Prepare to restructure after processing, if necessary
            if self.outfile_structure != 'site':
                processed_file = f'{self.outfile}.site.tmp'
                self._temp_files.append(processed_file)
            else:
                processed_file = self.outfile

            postprocessing_logger.info(f'Converting file {file_to_process} --> {processed_file}')

            # First we want to check if any records have all been done, to lighten our workload
            finished_records = set()
            if os.path.isfile(processed_file) and not kwargs.get('force', False):
                with h5py.File(processed_file, 'r') as f:
                    finished_records = set(f.keys())

            # Load file
            with h5py.File(file_to_process, 'r') as f:
                all_records = sorted(list(f.keys()))
                records_per_process = kwargs.get('avg_num', 1)      # Records are getting averaged together.

                if not kwargs.get('force', False):  # file may be partially processed, only process remaining records
                    final_records_remaining = sorted(list(
                        set(all_records[::records_per_process]).difference(finished_records)))

                first_idx = all_records.index(final_records_remaining[0])   # first record to process
                num_to_process = round(len(all_records) / records_per_process)
                num_completed = first_idx
                indices = range(first_idx, len(all_records), records_per_process)

                with get_context("spawn").Pool(kwargs.get('num_processes', 5)) as p:

                    function_to_call = partial(processing_machine,
                                               filename=file_to_process, record_keys=all_records,
                                               records_per_process=records_per_process,
                                               averaging_method=self.averaging_method,
                                               processing_fn=self.process_record, **kwargs)

                    for completed_record, i in p.imap(function_to_call, indices):
                        num_completed += 1
                        if completed_record is not None:
                            # Save record to temporary file
                            tempfile = f'/tmp/{all_records[i]}.tmp'
                            dd.io.save(tempfile, completed_record, compression=None)

                            # Copy record to output file
                            cmd = f'h5copy -i {tempfile} -o {processed_file} -s / -d {all_records[i]}'
                            sp.call(cmd.split())

                            # Remove temporary file
                            os.remove(tempfile)
                        completion_percentage = num_completed / num_to_process
                        bar_width = 60  # arbitrary width
                        filled = int(bar_width * completion_percentage)
                        unfilled = bar_width - filled
                        print(f'\r[{"="*filled}{" "*unfilled}] {completion_percentage*100:.2f}%', flush=True, end='')
                    print()     # Keep the progress bar on its own line

            # Restructure to final structure format, if necessary
            if self.outfile_structure != 'site':
                postprocessing_logger.info(f'Restructuring file {processed_file} --> {self.outfile}')
                restructure(processed_file, self.outfile, self.outfile_type, 'site', self.outfile_structure)
        except (Exception,) as e:
            postprocessing_logger.error(f'Could not process file {self.infile} -> {self.outfile}. Removing all newly'
                                        f' generated files.')
            postprocessing_logger.error(e)
        finally:
            self._remove_temp_files()

    def _remove_temp_files(self):
        """
        Deletes all temporary files used in the processing chain.
        """
        for filename in self._temp_files:
            if os.path.exists(filename):
                os.remove(filename)

    @staticmethod
    def process_record(record: OrderedDict, averaging_method: Union[None, str], **kwargs) -> OrderedDict:
        """
        This method should be overwritten by child classes, and should contain the necessary
        steps to process a record of input type to output type.

        Parameters
        ----------
        record: OrderedDict
            An hdf5 group containing one record of site-structured data
        averaging_method: Union[None, str]
            Method to use for averaging correlations across sequences.

        Returns
        -------
        record: OrderedDict
            The same hdf5 group, but with the necessary modifications to conform to the standard
            of data for self.final_type.
        """
        return record
