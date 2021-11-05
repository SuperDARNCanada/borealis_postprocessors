# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains functions for opening and converting
SuperDARN data files.
"""

import argparse
import os
import pydarnio
from exceptions import conversion_exceptions
from data_processing import convert_antennas_iq, convert_bfiq

SUPPORTED_FILE_TYPES = [
    'antennas_iq',
    'bfiq',
    'rawacf'
]

SUPPORTED_FILE_STRUCTURES = [
    'array',
    'site',
    'dmap'
]

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
    parser.add_argument("--final_type", required=True,
                        help="Type of output file. Acceptable types are 'antennas_iq', 'bfiq', and 'rawacf'.")
    parser.add_argument("--file_structure", required=True,
                        help="Structure of input file. Acceptable structures are "
                             "'array', 'site', and 'dmap' (dmap for rawacf type only).")
    parser.add_argument("--final_structure", required=True,
                        help="Structure of output file. Acceptable structures are 'array', 'site', "
                             "and 'dmap' (dmap for rawacf type only).")
    parser.add_argument("--averaging_method", required=False, default='mean',
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
        If not provided, it will try 'array' first, then 'site',
        then 'dmap'.
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
        if self.file_type not in FILE_TYPE_MAPPING.keys():
            raise conversion_exceptions.ImproperFileTypeError(
                'Input file type "{}" not supported. Supported types '
                'are {}'
                ''.format(self.file_type, FILE_TYPE_MAPPING.keys())
            )
        if self.final_type not in FILE_TYPE_MAPPING.keys():
            raise conversion_exceptions.ImproperFileTypeError(
                'Output file type "{}" not supported. Supported types '
                'are {}'
                ''.format(self.final_type, SUPPORTED_FILE_TYPES)
            )
        if self.file_structure not in FILE_STRUCTURE_MAPPING[self.file_type]:
            raise conversion_exceptions.ImproperFileStructureError(
                'Input file structure "{structure}" is not compatible with '
                'input file type "{type}": Valid structures for {type} are '
                '{valid}'.format(structure=self.file_structure,
                                 type=self.file_type,
                                 valid=FILE_STRUCTURE_MAPPING[self.file_type])
            )
        if self.final_type not in FILE_TYPE_MAPPING[self.file_type]:
            raise conversion_exceptions.ConversionUpstreamError(
                'Conversion from {filetype} to {final_type} is '
                'not supported. Only downstream processing is '
                'possible. Downstream types for {filetype} are'
                '{downstream}'.format(filetype=self.file_type,
                                      final_type=self.final_type,
                                      downstream=FILE_TYPE_MAPPING[self.final_type])
            )
        if self.final_structure not in FILE_STRUCTURE_MAPPING[self.final_type]:
            raise conversion_exceptions.ImproperFileStructureError(
                'Output file structure "{structure}" is not compatible with '
                'output file type "{type}": Valid structures for {type} are '
                '{valid}'.format(structure=self.final_structure,
                                 type=self.final_type,
                                 valid=FILE_STRUCTURE_MAPPING[self.final_type])
            )
        if self.file_type == self.final_type and self.file_structure == self.final_structure:
            raise conversion_exceptions.NoConversionNecessaryError(
                'Desired output format is same as input format.'
            )

        self._converter = self.get_converter()

    def _remove_temp_files(self):
        """
        Deletes all temporary files used in the conversion chain.
        """
        for filename in self._temp_files:
            os.remove(filename)

    def get_converter(self):
        """
        Determines the correct class of converter to instantiate based on the attributes
        of the ConvertFiles object.
        :return: Instantiated converter object
        """
        # TODO: Add arguments to these once the classes are created
        if self.file_type == 'antennas_iq':
            return ConvertAntennasIQ()
        elif self.file_type == 'bfiq':
            return ConvertBfiq()
        else:
            return ConvertRawacf()


def restructure_file(self, filename: str, output_file: str, file_type: str, file_structure: str, final_structure: str):
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


def antiq2bfiq(filename: str, output_file: str, file_structure: str, final_structure: str):
    """
    Converts an antennas_iq file to bfiq file.

    :param filename:            Name of antennas_iq file. String
    :param output_file:         Name of bfiq file. String
    :param file_structure:      Structure of antennas_iq file. String
    :param final_structure:     Structure of bfiq file. String
    :return:
    """
    file_type = 'antennas_iq'
    final_type = 'bfiq'
    temp_files = []

    # Convert array files to site files for processing
    if file_structure == 'array':
        site_file = '/tmp/tmp.antennas_iq'
        temp_files.append(site_file)
        restructure_file(filename, site_file, file_type, file_structure, 'site')
    else:
        site_file = filename

    # Determine name for the bfiq file
    if final_structure == 'site':
        bfiq_file = output_file
    else:
        bfiq_file = '/tmp/tmp.bfiq'
        temp_files.append(bfiq_file)

    # Convert antennas_iq.site file to bfiq.site file
    try:
        antennas_iq_to_bfiq.antennas_iq_to_bfiq(site_file, bfiq_file)
    except Exception:
        # Remove any temporary files before halting
        remove_temp_files(temp_files)
        raise

    # Restructure the file to final_structure
    restructure_file(bfiq_file, output_file, final_type, 'site', final_structure)

    # Remove any temporary files made along the way
    remove_temp_files(temp_files)


def bfiq2rawacf(filename: str, output_file: str, file_structure: str, final_structure: str,
                averaging_method: str = 'mean'):
    """
    Converts a bfiq file into a rawacf file.

    :param filename:            Name of the bfiq file. String
    :param output_file:         Name of the rawacf file. String
    :param file_structure:      Structure of the bfiq file. String
    :param final_structure:     Structure of the rawacf file. String
    :param averaging_method:    Averaging method for generating correlations. String
    """
    file_type = 'bfiq'
    final_type = 'rawacf'
    temp_files = []

    # Convert array files to site files for processing
    if file_structure == 'array':
        site_file = '/tmp/tmp.bfiq'
        temp_files.append(site_file)
        restructure_file(filename, site_file, file_type, file_structure, 'site')
    else:
        site_file = filename

    # Determine name for the bfiq file
    if final_structure == 'site':
        rawacf_file = output_file
    else:
        rawacf_file = '/tmp/tmp.bfiq'
        temp_files.append(rawacf_file)

    # Convert antennas_iq.site file to bfiq.site file
    try:
        bfiq_to_rawacf.bfiq_to_rawacf(site_file, rawacf_file, averaging_method=averaging_method)
    except Exception:
        # Remove any temporary files before halting
        remove_temp_files(temp_files)
        raise

    # Restructure the file to its final structure
    restructure_file(rawacf_file, output_file, final_type, 'site', final_structure)

    # Remove any temporary files made along the way
    remove_temp_files(temp_files)


def antiq2rawacf(filename: str, output_file: str, file_type: str, final_type: str,
                 file_structure: str, final_structure: str, averaging_method: str = 'mean'):

    # Set up intermediate bfiq file
    bfiq_file = '/tmp/tmp.bfiq'
    bfiq_structure = 'site'
    temp_files = [bfiq_file]

    # Convert to bfiq file
    try:
        antiq2bfiq(filename, bfiq_file, file_structure, bfiq_structure)

        # Convert bfiq file to rawacf file
        bfiq2rawacf(bfiq_file, output_file, bfiq_structure, final_structure, averaging_method=averaging_method)
    except Exception:
        # Remove temporary files before halting.
        remove_temp_files(temp_files)
        raise

    # Remove bfiq file
    remove_temp_files(temp_files)


def main():
    parser = conversion_parser()
    args = parser.parse_args()

    ConvertFile(args.infile, args.outfile, args.filetype, args.final_type, args.file_structure, args.final_structure,
                 averaging_method=args.averaging_method)


if __name__ == "__main__":
    main()
