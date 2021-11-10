# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains functions for opening and converting
SuperDARN data files.
"""

import argparse
import os

from data_processing.antennas_iq_to_bfiq import ProcessAntennasIQ2Bfiq
from data_processing.antennas_iq_to_rawacf import ProcessAntennasIQ2Rawacf
from data_processing.convert_base import BaseConvert
from data_processing.bfiq_to_rawacf import ProcessBfiq2Rawacf
from exceptions import conversion_exceptions


def usage_msg():
    """
    Return the usage message for this process.
    This is used if a -h flag or invalid arguments are provided.
    :returns: the usage message
    """

    usage_message = """ conversion.py [-h] infile outfile file_type final_type file_structure final_structure [averaging_method]
    
    Pass in the filename you wish to convert, the filename you wish to save as, and the types and structures of both.
    The script will :
    1. convert the input file into an output file of type "final_type" and structure "final_structure". If
       the final type is rawacf, the averaging method may optionally be specified as well (default is mean). """

    return usage_message


def conversion_parser():
    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("--infile", required=True,
                        help="Path to the file that you wish to convert. (e.g. 20190327.2210.38.sas.0.bfiq.hdf5.site)")
    parser.add_argument("--outfile", required=True,
                        help="Path to the location where the output file should be stored. "
                             "(e.g. 20190327.2210.38.sas.0.rawacf.hdf5.site)")
    parser.add_argument("--filetype", required=True,
                        help="Type of input file. Acceptable types are 'antennas_iq', 'bfiq', and 'rawacf'.")
    parser.add_argument("--final-type", required=True,
                        help="Type of output file. Acceptable types are 'antennas_iq', 'bfiq', and 'rawacf'.")
    parser.add_argument("--file-structure", required=True,
                        help="Structure of input file. Acceptable structures are "
                             "'array', 'site', and 'dmap' (dmap for rawacf type only).")
    parser.add_argument("--final-structure", required=True,
                        help="Structure of output file. Acceptable structures are 'array', 'site', "
                             "and 'dmap' (dmap for rawacf type only).")
    parser.add_argument("--averaging-method", required=False, default='mean',
                        help="Averaging method for generating rawacf type file. Allowed "
                             "methods are 'mean' (default) and 'median'.")

    return parser


class ConvertFile(object):
    """
    Class for general conversion of Borealis data files. This includes both restructuring of
    data files, and processing lower-level data files into higher-level data files. This class
    abstracts and redirects the conversion to the correct class (ConvertAntennasIQ, ConvertBfiq,
    ConvertRawacf).

    See Also
    --------
    ConvertAntennasIQ
    ConvertBfiq
    ConvertRawacf

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
    averaging_method: str
        Averaging method for computing correlations (for processing into rawacf files).
        Acceptable values are 'mean' and 'median'.
    """

    def __init__(self, filename: str, output_file: str, file_type: str, final_type: str,
                 file_structure: str, final_structure: str, averaging_method: str = 'mean'):
        self.filename = filename
        self.output_file = output_file
        self.file_type = file_type
        self.final_type = final_type
        self.file_structure = file_structure
        self.final_structure = final_structure
        self.averaging_method = averaging_method
        self._temp_files = []

        if not os.path.isfile(self.filename):
            raise conversion_exceptions.FileDoesNotExistError(
                'Input file {}'.format(self.filename)
            )

        self._converter = self.get_converter()

    def get_converter(self):
        """
        Determines the correct class of converter to instantiate based on the attributes
        of the ConvertFiles object.

        Returns
        -------
        Instantiated converter object
        """
        if self.final_type not in BaseConvert.type_mapping().keys():
            raise conversion_exceptions.ImproperFileTypeError(
                'Output file type "{}" not supported. Supported types '
                'are {}'
                ''.format(self.final_type, list(BaseConvert.type_mapping().keys()))
            )

        # Only restructuring necessary
        if self.file_type == self.final_type:
            if self.file_structure == self.final_structure:
                raise conversion_exceptions.NoConversionNecessaryError(
                    'Desired output format is same as input format.'
                )
            return BaseConvert.restructure(self.filename, self.output_file, self.file_type, self.file_structure,
                                           self.final_structure)

        if self.file_type == 'antennas_iq':
            if self.final_type == 'bfiq':
                return ProcessAntennasIQ2Bfiq(self.filename, self.output_file, self.file_structure,
                                              self.final_structure)
            elif self.final_type == 'rawacf':
                return ProcessAntennasIQ2Rawacf(self.filename, self.output_file, self.file_structure,
                                                self.final_structure, self.averaging_method)
            else:
                raise conversion_exceptions.ConversionUpstreamError(
                    'Conversion from {filetype} to {final_type} is '
                    'not supported. Only downstream processing is '
                    'possible. Downstream types for {filetype} are '
                    '{downstream}'.format(filetype=self.file_type,
                                          final_type=self.final_type,
                                          downstream=BaseConvert.type_mapping()[self.final_type])
                )
        elif self.file_type == 'bfiq':
            if self.final_type == 'rawacf':
                return ProcessBfiq2Rawacf(self.filename, self.output_file, self.file_structure, self.final_structure,
                                          self.averaging_method)
            else:
                raise conversion_exceptions.ConversionUpstreamError(
                    'Conversion from {filetype} to {final_type} is '
                    'not supported. Only downstream processing is '
                    'possible. Downstream types for {filetype} are '
                    '{downstream}'.format(filetype=self.file_type,
                                          final_type=self.final_type,
                                          downstream=BaseConvert.type_mapping()[self.final_type])
                )

        else:
            raise conversion_exceptions.ImproperFileTypeError(
                'Input file type "{}" not supported. Supported types '
                'are {}'
                ''.format(self.file_type, list(BaseConvert.type_mapping().keys()))
            )


def main():
    parser = conversion_parser()
    args = parser.parse_args()

    ConvertFile(args.infile, args.outfile, args.filetype, args.final_type, args.file_structure, args.final_structure,
                averaging_method=args.averaging_method)


if __name__ == "__main__":
    main()
