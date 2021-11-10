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

import numpy as np
import deepdish as dd
import pydarnio

try:
    import cupy as xp
except ImportError:
    import numpy as xp

    cupy_available = False
else:
    cupy_available = True

import logging

postprocessing_logger = logging.getLogger('borealis_postprocessing')


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
    filename: str
        The filename of the input file containing SuperDARN data.
    output_file: str
        The file name of output file
    file_type: str
        Type of data file. Types include:
        'antennas_iq'
        'bfiq'
        'rawacf'
    final_type: str
        Desired type of output data file. Same types as above.
    file_structure: str
        The write structure of the file. Structures include:
        'array'
        'site'
        'dmap'
        All borealis files are either 'site' or 'array' structured.
    final_structure: str
        The desired structure of the output file. Same structures as
        above.
    """
    def __init__(self, filename: str, output_file: str, file_type: str, final_type: str, file_structure: str,
                 final_structure: str):
        """
        Initializes the attributes of the class.

        Parameters
        ----------
        filename: str
            Path to the input file
        output_file: str
            Path to the output file
        file_type: str
            Borealis file type of input file. Supported types are:
            'antennas_iq'
            'bfiq'
            'rawacf'
        final_type: str
            Borealis file type of output file. Supported types are same as for file_type.
        file_structure: str
            Borealis file structure of input file. Supported structures are:
            'array'
            'site'
        final_structure: str
            Borealis file structure of output file. Supported structures are same as file_structure,
            with the addition of 'dmap' for rawacf-type files.
        """
        self.filename = filename
        self.output_file = output_file
        self.file_type = file_type
        self.file_structure = file_structure
        self.final_type = final_type
        self.final_structure = final_structure
        self.averaging_method = None
        self._temp_files = []

    def process_file(self):
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
        # Restructure to 'site' format if necessary
        if self.file_structure != 'site':
            file_to_process = '{}.site.tmp'.format(self.filename)
            self._temp_files.append(file_to_process)
            # Restructure file to site format for processing
            postprocessing_logger.info('Restructuring file {} --> {}'.format(self.filename, file_to_process))
            self.restructure(self.filename, file_to_process, self.file_type, self.file_structure, 'site')
        else:
            file_to_process = self.filename

        # Prepare to restructure after processing, if necessary
        if self.final_structure != 'site':
            processed_file = '{}.site.tmp'.format(self.output_file)
            self._temp_files.append(processed_file)
        else:
            processed_file = self.output_file

        postprocessing_logger.info('Converting file {} --> {}'.format(file_to_process, processed_file))
        
        # Load file
        group = dd.io.load(file_to_process)
        records = group.keys()

        # Process each record
        for record in records:
            record_dict = group[record]
            beamformed_record = self.process_record(record_dict, self.averaging_method)

            # Convert to numpy arrays for saving to file with deepdish
            formatted_record = self.convert_to_numpy(beamformed_record)

            # Save record to temporary file
            tempfile = '/tmp/{}.tmp'.format(record)
            dd.io.save(tempfile, formatted_record, compression=None)

            # Copy record to output file
            cmd = 'h5copy -i {} -o {} -s {} -d {}'
            cmd = cmd.format(tempfile, processed_file, '/', '/{}'.format(record))
            sp.call(cmd.split())

            # Remove temporary file
            os.remove(tempfile)

        # Restructure to final structure format, if necessary
        if self.final_structure != 'site':
            postprocessing_logger.info('Restructuring file {} --> {}'.format(processed_file, self.final_structure))
            self.restructure(processed_file, self.output_file, self.final_type, 'site', self.final_structure)

        self._remove_temp_files()

    def _remove_temp_files(self):
        """
        Deletes all temporary files used in the processing chain.
        """
        for filename in self._temp_files:
            os.remove(filename)

    @staticmethod
    def process_record(record: OrderedDict, averaging_method: Union[None, str]) -> OrderedDict:
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

    @staticmethod
    def restructure(infile_name, outfile_name, file_type, file_structure, final_structure):
        """
        This method restructures filename of structure "file_structure" into "final_structure".

        Parameters
        ----------
        infile_name: str
            Name of the original file.
        outfile_name: str
            Name of the restructured file.
        file_structure: str
            The current write structure of the file. One of 'array' or 'site'.
        final_structure: str
            The desired write structure of the file. One of 'array', 'site', 'iqdat', or 'dmap'.
        """
        # Dmap is handled specially, since it has its own function in pydarnio
        if final_structure == 'dmap':
            pydarnio.BorealisConvert(infile_name, file_type, outfile_name,
                                     borealis_file_structure=file_structure)
            return

        # Get data from the file
        reader = pydarnio.BorealisRead(infile_name, file_type, file_structure)

        # Get data in correct format for writing to output file
        if final_structure == 'site':
            data = reader.records
        else:
            data = reader.arrays

        # Write to output file
        pydarnio.BorealisWrite(outfile_name, data, file_type, final_structure)

    @staticmethod
    def convert_to_numpy(data):
        """Converts lists stored in dict into numpy array. Recursive.
        Args:
            data (Python dictionary): Dictionary with lists to convert to numpy arrays.
        """
        for k, v in data.items():
            if isinstance(v, dict):
                BaseConvert.convert_to_numpy(v)
            elif isinstance(v, list):
                data[k] = np.array(v)
            else:
                continue
        
        return data
