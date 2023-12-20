# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains functions for opening and converting
SuperDARN data files.
"""

import argparse
import os

from postprocessors import AntennasIQ2Bfiq, Bfiq2Rawacf, AntennasIQ2Rawacf, BaseConvert
from postprocessors.core.restructure import FILE_TYPE_MAPPING, restructure
from postprocessors import conversion_exceptions


def usage_msg():
    """
    Return the usage message for this process.
    This is used if a -h flag or invalid arguments are provided.
    """

    usage_message = """ conversion.py [-h] infile outfile infile_type outfile_type infile_structure outfile_structure
    
    Pass in the filename you wish to convert, the filename you wish to save as, and the types and structures of both.
    The script will convert the input file into an output file of type "outfile_type" and structure "outfile_structure".
    """

    return usage_message


def conversion_parser():
    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("infile",
                        help="Path to the file that you wish to convert. (e.g. 20190327.2210.38.sas.0.bfiq.hdf5.site)")
    parser.add_argument("outfile",
                        help="Path to the location where the output file should be stored. "
                             "(e.g. 20190327.2210.38.sas.0.rawacf.hdf5.site)")
    parser.add_argument("infile_type", metavar="infile-type", choices=['antennas_iq', 'bfiq', 'rawacf'],
                        help="Type of input file.")
    parser.add_argument("outfile_type", metavar="outfile-type", choices=['antennas_iq', 'bfiq', 'rawacf'],
                        help="Type of output file.")
    parser.add_argument("infile_structure", metavar="infile-structure", choices=['array', 'site'],
                        help="Structure of input file.")
    parser.add_argument("outfile_structure", metavar="outfile-structure", choices=['array', 'site', 'iqdat', 'dmap'],
                        help="Structure of output file.")

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
        The structure of the file. Structures include:
        'array'
        'site'
    outfile_structure: str
        The desired structure of the output file. Structures supported are:
        'array'
        'site'
        'iqdat' (bfiq only)
        'dmap' (rawacf only)
    """

    def __init__(self, infile: str, outfile: str, infile_type: str, outfile_type: str,
                 infile_structure: str, outfile_structure: str, **kwargs):
        self.infile = infile
        self.outfile = outfile
        self.infile_type = infile_type
        self.outfile_type = outfile_type
        self.infile_structure = infile_structure
        self.outfile_structure = outfile_structure
        self._temp_files = []

        if not os.path.isfile(self.infile):
            raise conversion_exceptions.FileDoesNotExistError(
                f'Input file {self.infile}'
            )

        self._converter = self.get_converter()
        if self._converter.__class__ != BaseConvert:
            self._converter.process_file(**kwargs)

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
                return AntennasIQ2Bfiq(self.infile, self.outfile, self.infile_structure, self.outfile_structure)
            elif self.outfile_type == 'rawacf':
                return AntennasIQ2Rawacf(self.infile, self.outfile, self.infile_structure, self.outfile_structure)
            else:
                raise conversion_exceptions.ConversionUpstreamError(
                    f'Conversion from {self.infile_type} to {self.outfile_type} is not supported. Only downstream '
                    f'processing is possible. Downstream types for {self.infile_type} are '
                    f'{FILE_TYPE_MAPPING[self.outfile_type]}'
                )
        elif self.infile_type == 'bfiq':
            if self.outfile_type == 'rawacf':
                return Bfiq2Rawacf(self.infile, self.outfile, self.infile_structure, self.outfile_structure)
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


def main():
    parser = conversion_parser()
    args = parser.parse_args()

    ConvertFile(args.infile, args.outfile, args.infile_type, args.outfile_type, args.infile_structure,
                args.outfile_structure)


if __name__ == "__main__":
    main()
