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

from data_processing.utils.restructure import restructure, convert_to_numpy, FILE_STRUCTURE_MAPPING
from exceptions import conversion_exceptions

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
                'Input file structure "{structure}" is not compatible with '
                'input file type "{type}": Valid structures for {type} are '
                '{valid}'.format(structure=self.infile_structure,
                                 type=self.infile_type,
                                 valid=FILE_STRUCTURE_MAPPING[self.infile_type])
            )
        if self.outfile_structure not in FILE_STRUCTURE_MAPPING[self.outfile_type]:
            raise conversion_exceptions.ImproperFileStructureError(
                'Output file structure "{structure}" is not compatible with '
                'output file type "{type}": Valid structures for {type} are '
                '{valid}'.format(structure=self.outfile_structure,
                                 type=self.outfile_type,
                                 valid=FILE_STRUCTURE_MAPPING[self.outfile_type])
            )
        if self.infile_structure not in ['array', 'site']:
            raise conversion_exceptions.ConversionUpstreamError(
                'Input file structure "{}" cannot be reprocessed into any other ' 
                'format.'.format(self.infile_structure)
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
        # Restructure to 'site' format if necessary
        if self.infile_structure != 'site':
            file_to_process = '{}.site.tmp'.format(self.infile)
            self._temp_files.append(file_to_process)
            # Restructure file to site format for processing
            postprocessing_logger.info('Restructuring file {} --> {}'.format(self.infile, file_to_process))
            restructure(self.infile, file_to_process, self.infile_type, self.infile_structure, 'site')
        else:
            file_to_process = self.infile

        # Prepare to restructure after processing, if necessary
        if self.outfile_structure != 'site':
            processed_file = '{}.site.tmp'.format(self.outfile)
            self._temp_files.append(processed_file)
        else:
            processed_file = self.outfile

        postprocessing_logger.info('Converting file {} --> {}'.format(file_to_process, processed_file))
        
        # Load file
        group = dd.io.load(file_to_process)
        records = group.keys()

        # Process each record
        for record in records:
            record_dict = group[record]
            beamformed_record = self.process_record(record_dict, self.averaging_method, **kwargs)

            # Convert to numpy arrays for saving to file with deepdish
            formatted_record = convert_to_numpy(beamformed_record)

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
        if self.outfile_structure != 'site':
            postprocessing_logger.info('Restructuring file {} --> {}'.format(processed_file, self.outfile_structure))
            restructure(processed_file, self.outfile, self.outfile_type, 'site', self.outfile_structure)

        self._remove_temp_files()

    def _remove_temp_files(self):
        """
        Deletes all temporary files used in the processing chain.
        """
        for filename in self._temp_files:
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
