# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains functions for opening and converting
SuperDARN data files.
"""

import argparse
import os
import pydarnio

from data_processing.convert_antennas_iq import ProcessAntennasIQ2Bfiq
from data_processing.convert_bfiq import ProcessBfiq2Rawacf
from exceptions import conversion_exceptions

# Keys are valid input file types, values are lists of allowed
# output file types. A file of type 'key' can be processed into
# any type in 'value'.
FILE_TYPE_MAPPING = {
    'antennas_iq': ['antennas_iq', 'bfiq', 'rawacf'],
    'bfiq': ['bfiq', 'rawacf'],
    'rawacf': ['rawacf']
}

# Keys are valid input file types, and values are lists of
# supported file structures for the file type.
FILE_STRUCTURE_MAPPING = {
    'antennas_iq': ['site', 'array'],
    'bfiq': ['site', 'array'],
    'rawacf': ['site', 'array', 'dmap']
}


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
    parser.add_argument("--save-intermediate-files", action='store_true',
                        help='Save intermediate files generated in processing chain. E.g. if processing antennas_iq '
                             '-> rawacf, save the bfiq file generated in between.')

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
                 file_structure: str, final_structure: str, averaging_method: str = 'mean',
                 save_intermediate_files: bool = False):
        self.check_args(filename, file_type, final_type, file_structure, final_structure)
        self.filename = filename
        self.output_file = output_file
        self.file_type = file_type
        self.final_type = final_type
        self.file_structure = file_structure
        self.final_structure = final_structure
        self.averaging_method = averaging_method
        self.save_intermediate_files = save_intermediate_files
        self._temp_files = []

        self._file_names = self.temp_file_names()

        if self.file_type == 'array':
            self.restructure_file(self.filename, _, self.file_type, self.file_structure, 'site')

        if self.file_type == 'antennas_iq' and self.final_type != 'antennas_iq':
            self._antiq2bfiq_converter = ProcessAntennasIQ2Bfiq(self.file_names['antennas_iq.site'], self.file_names['bfiq.site'], 'bfiq',
                                                                'site', 'site')
        if self.file_type != 'rawacf' and self.final_type == 'rawacf':
            self._bfiq2rawacf_convert = ProcessBfiq2Rawacf(self.filename, self.output_file, self.final_type,
                                                           self.file_structure, self.final_structure,
                                                           self.averaging_method)

    def _remove_temp_files(self):
        """
        Deletes all temporary files used in the conversion chain.
        """
        if not self.save_intermediate_files:
            for filename in self._temp_files:
                os.remove(filename)

    @staticmethod
    def check_args(filename, file_type, final_type, file_structure, final_structure):
        if not os.path.isfile(filename):
            raise conversion_exceptions.FileDoesNotExistError(
                'Input file {}'.format(filename)
            )
        if file_type not in FILE_TYPE_MAPPING.keys():
            raise conversion_exceptions.ImproperFileTypeError(
                'Input file type "{}" not supported. Supported types '
                'are {}'
                ''.format(file_type, FILE_TYPE_MAPPING.keys())
            )
        if final_type not in FILE_TYPE_MAPPING.keys():
            raise conversion_exceptions.ImproperFileTypeError(
                'Output file type "{}" not supported. Supported types '
                'are {}'
                ''.format(final_type, FILE_TYPE_MAPPING.keys())
            )
        if file_structure not in FILE_STRUCTURE_MAPPING[file_type]:
            raise conversion_exceptions.ImproperFileStructureError(
                'Input file structure "{structure}" is not compatible with '
                'input file type "{type}": Valid structures for {type} are '
                '{valid}'.format(structure=file_structure,
                                 type=file_type,
                                 valid=FILE_STRUCTURE_MAPPING[file_type])
            )
        if final_type not in FILE_TYPE_MAPPING[file_type]:
            raise conversion_exceptions.ConversionUpstreamError(
                'Conversion from {filetype} to {final_type} is '
                'not supported. Only downstream processing is '
                'possible. Downstream types for {filetype} are'
                '{downstream}'.format(filetype=file_type,
                                      final_type=final_type,
                                      downstream=FILE_TYPE_MAPPING[final_type])
            )
        if final_structure not in FILE_STRUCTURE_MAPPING[final_type]:
            raise conversion_exceptions.ImproperFileStructureError(
                'Output file structure "{structure}" is not compatible with '
                'output file type "{type}": Valid structures for {type} are '
                '{valid}'.format(structure=final_structure,
                                 type=final_type,
                                 valid=FILE_STRUCTURE_MAPPING[final_type])
            )
        if file_type == final_type and file_structure == final_structure:
            raise conversion_exceptions.NoConversionNecessaryError(
                'Desired output format is same as input format.'
            )
        # TODO: Catch case where dmap file is input (can't convert dmap to hdf5)

    def get_converter(self):
        """
        Determines the correct class of converter to instantiate based on the attributes
        of the ConvertFiles object.
        :return: Instantiated converter object
        """
        # TODO: Add arguments to these once the classes are created
        if self.file_type == 'antennas_iq':
            return ProcessAntennasIQ2Bfiq(self.filename, self.output_file, self.final_type,
                                          self.file_structure, self.final_structure)
        elif self.file_type == 'bfiq':
            return ProcessBfiq2Rawacf(self.filename, self.output_file, self.final_type,
                                      self.file_structure, self.final_structure, self.averaging_method)
        else:
            return ConvertRawacf()

    def temp_file_names(self):
        file_name_dict = {'antennas_iq.array': 'antennas_iq.hdf5',
                          'antennas_iq.site': 'antennas_iq.hdf5.site',
                          'bfiq.array': 'bfiq.hdf5',
                          'bfiq.site': 'bfiq.hdf5.site',
                          'rawacf.array': 'rawacf.hdf5',
                          'rawacf.site': 'rawacf.hdf5.site',
                          'rawacf.dmap': 'rawacf.dmap',
                          f'{self.file_type}.{self.file_structure}': self.filename,
                          f'{self.final_type}.{self.final_structure}': self.output_file}

        return file_name_dict

    @staticmethod
    def restructure_file(filename: str, output_file: str, file_type: str, file_structure: str, final_structure: str):
        """
        Restructures a file, saving it as output_file.

        :param filename:        Name of input file. String
        :param output_file:     Name of output file. String
        :param file_type:       Type of input file. String
        :param file_structure:  Structure of input file. String
        :param final_structure: Structure of output file. String
        """

        # Dmap is handled specially, since it has its own function in pydarnio
        if final_structure == 'dmap':
            pydarnio.BorealisConvert(filename, file_type, output_file, borealis_file_structure=file_structure)
            return

        # Get data from the file
        reader = pydarnio.BorealisRead(filename, file_type, file_structure)

        # Get data in correct format for writing to output file
        if final_structure == 'site':
            data = reader.records
        else:
            data = reader.arrays

        # Write to output file
        pydarnio.BorealisWrite(output_file, data, file_type, final_structure)


def main():
    parser = conversion_parser()
    args = parser.parse_args()

    ConvertFile(args.infile, args.outfile, args.filetype, args.final_type, args.file_structure, args.final_structure,
                averaging_method=args.averaging_method, save_intermediate_files=args.save_intermediate_files)


if __name__ == "__main__":
    main()
