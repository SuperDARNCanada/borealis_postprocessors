# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains base functionality for postprocessing of Borealis data files.
"""
import os
import traceback
from collections import OrderedDict
from typing import Union
import h5py
from functools import partial
from multiprocessing import get_context

import postprocessors.core.restructure as rs
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


def processing_machine(idx: int, filename: str, record_keys: list, records_per_process: int, processing_fn,
                       file_type: str, version: tuple, **kwargs):
    """
    Helper function for processing a single record. It is defined here to facilitate multiprocessing.

    Parameters
    ----------
    idx: int
        Index into record_keys which tells processing_machine() which record to process
    filename: str
        HDF5 file with records to process.
    record_keys: list
        List of all top-level keys of the HDF5 file.
    records_per_process: int
        Number of records to process per call to this function.
    processing_fn: callable
        Function to call to process a record.
    file_type: str
        File type that is being processed. One of 'antennas_iq', 'bfiq', or 'rawacf'.
    version: tuple
        Version numbers of the record. (major, minor[, patch])
    kwargs: dict
        Key-word arguments to pass to processing_fn

    Returns
    -------
    formatted_record, idx: properly-formatted processed record and the index which was processed.
    """
    with h5py.File(filename, 'r') as hdf5_file:
        record_dict = rs.read_group(hdf5_file[record_keys[idx]], file_type)
        record_list = []  # List of all 'extra' records to process

        # If processing multiple records at a time, get all the records ready
        if records_per_process > 1:
            for num in range(idx + 1, min(idx + records_per_process, len(record_keys))):
                if version[0] > 0:
                    record_list.append(hdf5_file[record_keys[num]])
                else:
                    record_list.append(rs.read_group(hdf5_file[record_keys[num]], file_type))

    processed_record = processing_fn(record_dict, extra_records=record_list, **kwargs)

    if processed_record is None:
        return None, idx
    else:
        # Convert to numpy arrays for saving to file
        formatted_record = rs.convert_to_numpy(processed_record, version=version)
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
        The structure of the file. Structures include:
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

        if self.infile_structure not in rs.FILE_STRUCTURE_MAPPING[self.infile_type]:
            raise conversion_exceptions.ImproperFileStructureError(
                f'Input file structure "{self.infile_structure}" is not compatible with input file type '
                f'"{self.infile_type}": Valid structures for {self.infile_type} are '
                f'{rs.FILE_STRUCTURE_MAPPING[self.infile_type]}'
            )
        if self.outfile_structure not in rs.FILE_STRUCTURE_MAPPING[self.outfile_type]:
            raise conversion_exceptions.ImproperFileStructureError(
                f'Output file structure "{self.outfile_structure}" is not compatible with output file type '
                f'"{self.outfile_type}": Valid structures for {self.outfile_type} are '
                f'{rs.FILE_STRUCTURE_MAPPING[self.outfile_type]}'
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
        2. Apply appropriate downstream processing by calling process_record() on each record
        3. Restructure to final format
        4. Remove all intermediate files created along the way

        Parameters
        ----------
        **kwargs: dict
            Supported kwargs include:
                force: bool, if True will overwrite an existing output file
                avg_num: int, how many records are grouped together for a single process_record() call
                num_processes: int, how many CPU cores to distribute the job across
                keep_intermediate_files: bool, if True all intermediate files are not discarded
            Other kwargs may be supported by child classes and will be passed through to the process_record() function.
        """
        if os.path.isfile(self.outfile) and not kwargs.get('force', False):
            choice = input(f'Output file {self.outfile} already exists. Proceed anyway? Only records which don\'t '
                           f'exist in output file will be processed. (y/n): ')
            if choice[0] not in ['y', 'Y']:
                return 0

        version = self._get_version()
        try:
            # Restructure to 'site' format if necessary
            if self.infile_structure != 'site':
                if version[0] >= 1:
                    raise ValueError("All files after Borealis v1.0 are site-structured")
                file_to_process = f'{self.infile}.site'
                if not kwargs.get('keep_intermediate_files', False):
                    file_to_process += '.tmp'
                self._temp_files.append(file_to_process)
                # Restructure file to site format for processing
                postprocessing_logger.info(f'Restructuring file {self.infile} --> {file_to_process}')
                rs.restructure(self.infile, file_to_process, self.infile_type, self.infile_structure, 'site', version[0])
            else:
                file_to_process = self.infile

            # Prepare to restructure after processing, if necessary
            if self.outfile_structure != 'site':
                if version[0] < 1:
                    processed_file = f'{self.outfile}.site'
                elif self.outfile_structure != 'dmap':
                    raise ValueError("Cannot have array-structured Borealis v1.0+ file")
                else:
                    processed_file = f'{self.outfile}.h5'
                if not kwargs.get('keep_intermediate_files', False):
                    processed_file += '.tmp'
                self._temp_files.append(processed_file)
            else:
                processed_file = self.outfile

            postprocessing_logger.info(f'Converting file {file_to_process} --> {processed_file}')

            # First we want to check if any records have all been done, to lighten our workload
            finished_records = set()
            if os.path.isfile(processed_file) and not kwargs.get('force', False):
                with h5py.File(processed_file, 'r') as f:
                    finished_records = set(f.keys())

            # Load record names from file
            with h5py.File(file_to_process, 'r') as infile:
                all_records = sorted(list(infile.keys()))
                if version[0] >= 1:
                    all_records.remove("metadata")

            records_per_process = kwargs.get('avg_num', 1)      # Records getting averaged together.
            if not kwargs.get('force', False):      # file may be partially processed, only process remaining records
                final_records_remaining = sorted(list(
                    set(all_records[::records_per_process]).difference(finished_records)))
            else:
                final_records_remaining = all_records[::records_per_process]

            first_idx = all_records.index(final_records_remaining[0])   # first record to process
            num_to_process = round(len(all_records) / records_per_process)
            num_completed = first_idx
            indices = range(first_idx, len(all_records), records_per_process)

            function_to_call = partial(processing_machine,
                                       filename=file_to_process, record_keys=all_records,
                                       records_per_process=records_per_process,
                                       processing_fn=self.process_record, file_type=self.infile_type,
                                       version=version, **kwargs)

            # Do the processing on each record
            with h5py.File(processed_file, 'a') as outfile:
                def append_to_file(rec):
                    """Convenience function to append to file"""
                    if rec is not None:
                        rs.write_records(outfile, {all_records[i]: rec}, version=version)

                def progress_bar(done_so_far, total):
                    """Convenience function to print a progress bar"""
                    completion_percentage = done_so_far / total
                    bar_width = 60  # arbitrary width
                    filled = int(bar_width * completion_percentage)
                    unfilled = bar_width - filled
                    print(f'\r[{"=" * filled}{" " * unfilled}] {completion_percentage * 100:.2f}%', flush=True, end='')

                # Add the metadata to outfile first
                if version[0] >= 1:
                    with h5py.File(file_to_process, 'r') as infile:
                        metadata = rs.read_group(infile['metadata'], self.infile_type)
                        first_rec = rs.read_group(infile[all_records[0]], self.infile_type)
                    rs.write_records(outfile, {"metadata": metadata}, version=version)
                    self._update_metadata(first_rec, outfile["metadata"], **kwargs)
                    del first_rec  # only need it for getting all the correct metadata

                num_processes = kwargs.get("num_processes", 1)
                if num_processes > 1:   # Use multiprocessing if specified
                    with get_context("spawn").Pool(num_processes) as p:
                        for completed_record, i in p.imap(function_to_call, indices):
                            append_to_file(completed_record)
                            num_completed += 1
                            progress_bar(num_completed, num_to_process)
                else:   # Default single-worker
                    for idx in indices:
                        completed_record, i = function_to_call(idx)
                        append_to_file(completed_record)
                        num_completed += 1
                        progress_bar(num_completed, num_to_process)
                print('\r', flush=True, end='')     # Remove the progress bar

            # Restructure to final structure format, if necessary
            if self.outfile_structure != 'site':
                postprocessing_logger.info(f'Restructuring file {processed_file} --> {self.outfile}')
                rs.restructure(processed_file, self.outfile, self.outfile_type, 'site', self.outfile_structure, version[0])
        except (Exception,) as e:
            postprocessing_logger.error(f'Could not process file {self.infile} -> {self.outfile}. Removing all newly'
                                        f' generated files.')
            postprocessing_logger.error(e)
            postprocessing_logger.error(traceback.print_exc())
            raise e
        finally:
            if kwargs.get('keep_intermediate_files', False):
                self._temp_files = []
            else:
                self._remove_temp_files()

    def _remove_temp_files(self):
        """
        Deletes all temporary files used in the processing chain.
        """
        for filename in self._temp_files:
            if os.path.exists(filename):
                os.remove(filename)

    def _get_version(self):
        """
        Determines the version of Borealis that created the file

        Returns
        -------
        versions: [int]
            (major, minor[, patch]) version numbers
        """
        with h5py.File(self.infile, 'r') as f:
            if 'metadata' in f.keys():
                githash = f['metadata']['borealis_git_hash'][()]
            else:
                if 'borealis_git_hash' in f.attrs.keys():
                    githash = f.attrs['borealis_git_hash']
                else:
                    rec = sorted(list(f.keys()))[0]
                    githash = f[rec].attrs['borealis_git_hash']

            version = [int(i) for i in githash.decode('utf-8').split('-')[0].lstrip('v').split('.')]
        return version

    @classmethod
    def process_record(cls, record: OrderedDict, **kwargs) -> OrderedDict:
        """
        This method should be overwritten by child classes, and should contain the necessary
        steps to process a record of input type to output type.

        Parameters
        ----------
        record: OrderedDict
            An hdf5 group containing one record of site-structured data

        Returns
        -------
        record: OrderedDict
            The same hdf5 group, but with the necessary modifications to conform to the standard
            of data for self.final_type.
        """
        return record

    @classmethod
    def _update_metadata(cls, record: OrderedDict, metadata: h5py.Group, **kwargs):
        """
        Adds in metadata fields required for this file type.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing one averaging period worth of data and metadata
        metadata: h5py.Group
            metadata group in the output file
        kwargs: dict
            any other arguments that may be required (e.g. 'averaging_method' for rawacf generation)
        """
        return