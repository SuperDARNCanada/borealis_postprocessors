# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains functions for opening and converting
SuperDARN data files.
"""


import os

import pydarnio

from exceptions import conversion_exceptions

SUPPORTED_FILE_TYPES = [
    'antennas_iq',
    'bfiq',
    'rawacf'
]

SUPPORTED_FILE_STRUCTURES = [
    'array',
    'site',
    'rawacf'
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


def convert_file(filename: str, output_file: str, file_type: str, final_type: str,
                 file_structure: str = 'array', final_structure: str = 'array'):
    """
    Reads a SuperDARN data file, and converts it to the desired file
    type and structure.

    Parameters
    ----------
        filename: str
            file name containing SuperDARN data.
        output_file: str
            file name of output file
        file_type: str
            Type of data file. Types include:
            'antennas_iq'
            'bfiq'
            'rawacf'
        final_type:
            Desired type of output data file. Same types as above.
        file_structure:
            The write structure of the file. Structures include:
            'array'
            'site'
            'dmap'
            All borealis files are either 'site' or 'array' structured.
            If not provided, it will try 'array' first, then 'site',
            then 'dmap'.
        final_structure:
            The desired structure of the output file. Same structures as
            above.
    """
    if file_type not in SUPPORTED_FILE_TYPES:
        raise conversion_exceptions.ImproperFileTypeError(
            'Input file type {} not supported. Supported types'
            'are {}'
            ''.format(file_type, SUPPORTED_FILE_TYPES)
        )

    if file_structure not in SUPPORTED_FILE_STRUCTURES:
        raise conversion_exceptions.ImproperFileStructureError(
            'Input file structure {} not supported. Supported structures'
            'are {}'
            ''.format(file_structure, SUPPORTED_FILE_STRUCTURES)
        )

    if final_type not in SUPPORTED_FILE_TYPES:
        raise conversion_exceptions.ImproperFileTypeError(
            'Output file type {} not supported. Supported types'
            'are {}'
            ''.format(final_type, SUPPORTED_FILE_TYPES)
        )

    if final_structure not in SUPPORTED_FILE_STRUCTURES:
        raise conversion_exceptions.ImproperFileStructureError(
            'Output file structure {} not supported. Supported structures'
            'are {}'
            ''.format(final_structure, SUPPORTED_FILE_STRUCTURES)
        )

    if file_structure not in FILE_STRUCTURE_MAPPING[file_type]:
        raise conversion_exceptions.ImproperFileStructureError(
            'Input file structure {structure} is not compatible with '
            'input file type {type}: Valid structures for {type} are '
            '{valid}'.format(structure=file_structure,
                             type=file_type,
                             valid=FILE_STRUCTURE_MAPPING[file_type])
        )
    
    if final_type not in FILE_TYPE_MAPPING[final_type]:
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
            'Output file structure {structure} is not compatible with '
            'output file type {type}: Valid structures for {type} are '
            '{valid}'.format(structure=final_structure,
                             type=final_type,
                             valid=FILE_STRUCTURE_MAPPING[final_type])
        )

    if file_type == final_type and file_structure == final_structure:
        raise conversion_exceptions.NoConversionNecessaryError(
            'Desired output format is same as input format.'
        )

    if not os.path.isfile(filename):
        raise conversion_exceptions.FileDoesNotExistError(
            'Input file {}'.format(filename)
        )

    # No downstream processing, only sideways conversion
    if file_type == final_type:
        if file_structure == 'dmap':
            raise conversion_exceptions.ConversionUpstreamError(
                'Cannot convert upstream from dmap structure. '
                'Dmap format can only be the final file '
                'structure.'
            )
        # Write the data to SuperDARN DMap file
        if final_structure == 'dmap':
            converter = pydarnio.BorealisConvert(filename,
                                                 file_type,
                                                 output_file,
                                                 0,
                                                 borealis_file_structure=file_structure)
        # Converting between Borealis file structures
        else:
            reader = pydarnio.BorealisRead(filename,
                                           file_type,
                                           file_structure)
            if file_structure == 'array':
                data = reader.arrays
            else:   # site structured
                data = reader.records
            writer = pydarnio.BorealisWrite(output_file,
                                            data,
                                            final_type,
                                            final_structure)
    # Downstream processing necessary
    else:
        # TODO: Implement downstream processing
        pass
