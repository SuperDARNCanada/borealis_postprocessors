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
            if self.infile_structure not in ['array', 'site', 'dmap']:
                raise conversion_exceptions.ConversionUpstreamError(
                    f'Input file structure "{self.infile_structure}" cannot be reprocessed into any other format.'
                )
        check_args(self)
        self.averaging_method = None
        self._temp_files = []
        self.process_file(force = True)

    def binTimestamps(self, file, filetype): #sort timestamps to a corresponding beam number/index from 0-16
        with h5py.File(file, 'r') as f:
            timestamps = list(f.keys())
            indices = dict()
            timestamps.sort()
            cnt = 0
            for i in timestamps:
                if cnt < 16:
                    indices[i] = cnt
                    cnt += 1
                else:
                    cnt = 0
                    indices[i] = cnt
                    cnt += 1
        return indices
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


        if (self.infile_structure == 'dmap') and (self.infile_type == 'rawacf'): #Dmap input
            file_to_process = self.infile
            processed_file = self.outfile
            sdarn_read = pydarnio.SDarnRead(file_to_process)
            postprocessing_logger.info(f'converting file {file_to_process} --> {processed_file}')
            data = sdarn_read.read_rawacf()
            record = dict()
            all_records = [] #record names
            for rec in data: #Find the record names
                all_records.append(str(rec['time.yr']) + str(rec['time.mo']) + str(rec['time.dy']) + str(rec['time.hr']) + str(rec['time.mt']) + str(rec['time.sc']) + str(rec['time.us']))
            all_records = np.unique(all_records)
            for i in all_records: #reformat the records in 16 records per entry to better visualise FullFOV
                beam_rec = []
                for rec in data:
                    rec_time = str(rec['time.yr']) + str(rec['time.mo']) + str(rec['time.dy']) + str(
                        rec['time.hr']) + str(
                        rec['time.mt']) + str(rec['time.sc']) + str(rec['time.us'])
                    if rec_time == i:
                        beam_rec.append(rec)
                record[i] = beam_rec
            beamedrec = self.beam_process_2_normal(all_records, record)
            pydarnio.SDarnWrite(beamedrec, processed_file).write_rawacf(processed_file)
        else:
            version = self._get_version()
            # version = [0.7, 0.7]
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
                    rs.restructure(self.infile, file_to_process, self.infile_type, self.infile_structure, 'site',
                                   version[0])
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

                records_per_process = kwargs.get('avg_num', 1)  # Records getting averaged together.
                if not kwargs.get('force', False):  # file may be partially processed, only process remaining records
                    final_records_remaining = sorted(list(
                        set(all_records[::records_per_process]).difference(finished_records)))
                else:
                    final_records_remaining = all_records[::records_per_process]

                first_idx = all_records.index(final_records_remaining[0])  # first record to process
                num_to_process = round(len(all_records) / records_per_process)
                num_completed = first_idx
                indices = range(first_idx, len(all_records), records_per_process)
                beam_index = self.binTimestamps(file_to_process,self.infile_type) #Find beam indices associated with timestamps
                kwargs['beam_index'] = beam_index
                # Do the processing on each record
                with h5py.File(processed_file, 'a') as outfile:
                    def append_to_file(rec, idx):
                        """Convenience function to append to file"""
                        if rec is not None:
                            rs.write_records(outfile, {all_records[idx]: rec}, version=version)

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
                        kwargs['metadata'] = outfile['metadata']
                        del first_rec  # only need it for getting all the correct metadata

                    function_to_call = partial(cb.processing_machine,
                                               filename=file_to_process, record_keys=all_records,
                                               records_per_process=records_per_process,
                                               processing_fn=self.process_record, file_type=self.infile_type,
                                               version=version, **kwargs)

                    num_processes = kwargs.get("num_processes", 1)
                    if num_processes > 1:  # Use multiprocessing if specified
                        with get_context("spawn").Pool(num_processes) as p:
                            for completed_record, i in p.imap(function_to_call, indices):
                                append_to_file(completed_record, i)
                                num_completed += 1
                                progress_bar(num_completed, num_to_process)
                    else:  # Default single-worker
                        for idx in indices:
                            completed_record, i = function_to_call(idx)
                            append_to_file(completed_record, i)
                            num_completed += 1
                            progress_bar(num_completed, num_to_process)
                    print('\r', flush=True, end='')  # Remove the progress bar

                # Restructure to final structure format, if necessary
                if self.outfile_structure != 'site':
                    postprocessing_logger.info(f'Restructuring file {processed_file} --> {self.outfile}')
                    rs.restructure(processed_file, self.outfile, self.outfile_type, 'site', self.outfile_structure,
                                   version[0])
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

    @staticmethod
    def process_record(record: OrderedDict, **kwargs) -> OrderedDict:
        """
        Takes a record from an rawacf file process into a rawacf record.
        This method also keeps an single designated beam,

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing antennas_iq data and metadata
        beam_num: Union[None, str]
            Method to use for averaging correlations across sequences. Acceptable methods are 'median' and 'mean'

        Returns
        -------
        record: OrderedDict
            hdf5 record, with new fields required by rawacf data format
        """
        beam_index = kwargs.get('beam_index', None)
        first_timestamp = int(record['sqn_timestamps'][0]*1000)
        beamkeys = list(beam_index.keys())
        index = np.argmin(abs(first_timestamp - np.array([int(i) for i in beamkeys])))
        beam2keep = beam_index[beamkeys[index]]
        record['beam_nums'] = np.array([np.uint32(beam2keep)])
        record['beam_azms'] = np.array([record['beam_azms'][beam2keep]])
        try:
            record['data_dimensions'][0] = 1
        except:
            record['correlation_dimensions'][0] = 1
        record['main_acfs'] = record['main_acfs'][beam2keep, :, :].reshape(1, record['data_dimensions'][1], record['data_dimensions'][2])
        record['intf_acfs'] = record['intf_acfs'][beam2keep, :, :].reshape(1, record['data_dimensions'][1], record['data_dimensions'][2])
        record['xcfs'] = record['xcfs'][beam2keep, :, :].reshape(1, record['data_dimensions'][1], record['data_dimensions'][2])
        if beam2keep == 0:
            record['scan_start_marker'] = True
        else:
            record['scan_start_marker'] = False
        return record

    @staticmethod
    def beam_process_2_normal(all_records: list, record: dict, **kwargs) -> dict:
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
        cnt = 0  # initialize beam counter
        beamedrec = []
        for i in all_records:
            if cnt < 16:
                newrec = record[i][cnt]
                if cnt == 0:
                    newrec['scan'] = np.int16(1)
                else:
                    newrec['scan'] = np.int16(0)
                cnt += 1
            else:
                cnt = 0
                newrec = record[i][cnt]
                if cnt == 0:
                    newrec['scan'] = np.int16(1)
                else:
                    newrec['scan'] = np.int16(0)
                cnt += 1
            beamedrec.append(newrec)
        return beamedrec
