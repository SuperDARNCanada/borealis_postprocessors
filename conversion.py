# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains functions for opening and converting
SuperDARN data files.
"""

import argparse
import os
import glob

from data_processing.antennas_iq_to_bfiq import ProcessAntennasIQ2Bfiq
from data_processing.antennas_iq_to_rawacf import ProcessAntennasIQ2Rawacf
from data_processing.bfiq_to_rawacf import ProcessBfiq2Rawacf
from data_processing.convert_base import BaseConvert
from data_processing.utils.restructure import FILE_TYPE_MAPPING, restructure
from exceptions import conversion_exceptions


def usage_msg():
    """
    Return the usage message for this process.
    This is used if a -h flag or invalid arguments are provided.
    """

    usage_message = """ conversion.py [-h] infile outfile infile_type outfile_type infile_structure outfile_structure [--averaging-method a]
    
    Pass in the filename you wish to convert, the filename you wish to save as, and the types and structures of both.
    The script will convert the input file into an output file of type "outfile_type" and structure "outfile_structure".
    If the final type is rawacf, the averaging method may optionally be specified as well. """

    return usage_message


def conversion_parser():
    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("infile",
                        help="Path to the file that you wish to convert. (e.g. 20190327.2210.38.sas.0.bfiq.hdf5.site)")
    parser.add_argument("outfile",
                        help="Path to the location where the output file should be stored. "
                             "(e.g. 20190327.2210.38.sas.0.rawacf.hdf5.site)")
    parser.add_argument("infile_type", choices=['antennas_iq', 'bfiq', 'rawacf'],
                        help="Type of input file.")
    parser.add_argument("outfile_type", choices=['antennas_iq', 'bfiq', 'rawacf'],
                        help="Type of output file.")
    parser.add_argument("infile_structure", choices=['array', 'site'],
                        help="Structure of input file.")
    parser.add_argument("outfile_structure", choices=['array', 'site', 'iqdat', 'dmap'],
                        help="Structure of output file.")
    parser.add_argument("-a", "--averaging-method", required=False, default='mean', choices=['mean', 'median'],
                        help="Averaging method for generating rawacf type file. Default mean.")
    parser.add_argument("-m", "--multiprocessing", action="store_true", help="Enable multicore processing for batch jobs.")
    parser.add_argument('-i', '--indirectory', type=str, help='directory containing ZVH files with data to be plotted.')
    parser.add_argument('-o', '--outdirectory', type=str, default='', help='directory to save output plots.')
    parser.add_argument('-p', '--pattern', type=str, help='the file naming pattern less the appending numbers.')
    parser.add_argument('-v', '--verbose', action='store_true', help='explain what is being done verbosely.')
    return parser


class ConvertFile(object):
    """
    Class for general conversion of Borealis data files. This includes both restructuring of
    data files, and processing lower-level data files into higher-level data files. This class
    abstracts and redirects the conversion to the correct class (ProcessAntennasIQ2Bfiq,
    ProcessBfiq2Rawacf, and ProcessAntennasIQ2Rawacf).

    See Also
    --------
    ProcessAntennasIQ2Bfiq
    ProcessAntennasIQ2Rawacf
    ProcessBfiq2Rawacf
    BaseConvert

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
    outfile_structure: str
        The desired structure of the output file. Structures supported are:
        'array'
        'site'
        'iqdat' (bfiq only)
        'dmap' (rawacf only)
    averaging_method: str
        Averaging method for computing correlations (for processing into rawacf files).
        Acceptable values are 'mean' and 'median'.
    """

    def __init__(self, infile: str, outfile: str, infile_type: str, outfile_type: str,
                 infile_structure: str, outfile_structure: str, averaging_method: str = 'mean'):
        self.infile = infile
        self.outfile = outfile
        self.infile_type = infile_type
        self.outfile_type = outfile_type
        self.infile_structure = infile_structure
        self.outfile_structure = outfile_structure
        self.averaging_method = averaging_method
        self._temp_files = []

        if not os.path.isfile(self.infile):
            raise conversion_exceptions.FileDoesNotExistError(
                f'Input file {self.infile}'
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
        if self.outfile_type not in FILE_TYPE_MAPPING.keys():
            raise conversion_exceptions.ImproperFileTypeError(
                f'Output file type "{self.outfile_type}" not supported. Supported types '
                f'are {list(FILE_TYPE_MAPPING.keys())}'
            )

        # Only restructuring necessary
        if self.infile_type == self.outfile_type:
            if self.infile_structure == self.outfile_structure:
                raise conversion_exceptions.NoConversionNecessaryError(
                    'Desired output format is same as input format.'
                )
            # Restructure file, then return BaseConvert object for consistency
            restructure(self.infile, self.outfile, self.infile_type, self.infile_structure, self.outfile_structure)
            return BaseConvert(self.infile, self.outfile, self.infile_type, self.outfile_type, self.infile_structure,
                               self.outfile_structure)

        if self.infile_type == 'antennas_iq':
            if self.outfile_type == 'bfiq':
                return ProcessAntennasIQ2Bfiq(self.infile, self.outfile, self.infile_structure,
                                              self.outfile_structure)
            elif self.outfile_type == 'rawacf':
                return ProcessAntennasIQ2Rawacf(self.infile, self.outfile, self.infile_structure,
                                                self.outfile_structure, self.averaging_method)
            else:
                raise conversion_exceptions.ConversionUpstreamError(
                    f'Conversion from {self.infile_type} to {self.outfile_type} is not supported. Only downstream '
                    f'processing is possible. Downstream types for {self.infile_type} are '
                    f'{FILE_TYPE_MAPPING[self.outfile_type]}'
                )
        elif self.infile_type == 'bfiq':
            if self.outfile_type == 'rawacf':
                return ProcessBfiq2Rawacf(self.infile, self.outfile, self.infile_structure, self.outfile_structure,
                                          self.averaging_method)
            else:
                raise conversion_exceptions.ConversionUpstreamError(
                    f'Conversion from {self.infile_type} to {self.outfile_type} is not supported. Only downstream '
                    f'processing is possible. Downstream types for {self.infile_type} are '
                    f'{FILE_TYPE_MAPPING[self.outfile_type]}'
                )

        else:
            raise conversion_exceptions.ImproperFileTypeError(
                f'Input file type "{self.infile_type}" not supported. Supported types are '
                f'{list(FILE_TYPE_MAPPING.keys())}'
            )


def multiprocessing_conversion(arguments):
    """
    Parameters
    ----------
        arguments : list[tuples]
            A list of tuples which are the arguments for the function to be run.

    Returns
    -------
        output :
            A list of the returns from the function ran. May not be ordered.
    """
    import multiprocessing as mp

    # Todo: Use less than all the cores or give user options
    pool = mp.Pool(mp.cpu_count())
    pool.map(ConvertFile, arguments)

    return


def get_batch(directory, pattern='*', verbose=False):
    """
    Parameters
    ----------
        directory : str
            The directory or parent directory containing the .csv files.
        pattern : str
            The file naming pattern of files to load; eg. rkn_vswr would yield all rkn_vswr*.csv in directory tree.
        verbose : bool
            True will print more information about whats going on, False squelches.

    Returns
    -------
    """
    files = glob.glob(directory + '/**/' + pattern, recursive=True)  # + pattern
    if files == []:
        files = glob.glob(directory + pattern)
        if files == []:
            print(f'no files found at: {directory} or')
            print(f'no files found with pattern: {pattern}')
            exit()
    verbose and print("files found:\n", files)

    return files


def parse_filename(in_filename, intype, outtype, instruct, outstruct):
    file = in_filename.split('.')
    for n, f in enumerate(file):
        if f in intype:
            file[n] = outtype
        # else:
        #     print(f"infile_type: {intype} does not match file name {in_filename}")
        #     exit()
        if f in instruct:
            file[n] = outstruct
        # else:
        #     print(f"infile_type: {instruct} does not match file name {in_filename}")
        #     exit()
    file = '.'.join(file)
    print('outfile name:', file)
    return file


if __name__ == "__main__":
    parser = conversion_parser()
    args = parser.parse_args()
    if args.multiprocessing:
        files = get_batch(args.indirectory, args.pattern, args.verbose)
        arguments = []
        for file in files:
            outfile = parse_filename(file, args.infile_type, args.outfile_type,
                                     args.infile_structure, args.outfile_structure)
            arguments_tuple = (file, outfile, args.infile_type, args.outfile_type,
                               args.infile_structure, args.outfile_structure, args.averaging_method)
            arguments.append(arguments_tuple)

        multiprocessing_conversion(arguments)

    else:
        ConvertFile(args.infile, args.outfile, args.infile_type, args.outfile_type, args.infile_structure,
                    args.outfile_structure, averaging_method=args.averaging_method)
