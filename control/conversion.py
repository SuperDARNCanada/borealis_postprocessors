# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains functions for opening and converting
SuperDARN data files.
"""


import os
import subprocess as sp
import pydarnio
from exceptions import conversion_exceptions
from data_processing import antennas_iq_to_bfiq, bfiq_to_rawacf

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


def remove_temp_files(temp_file_list):
    """
    Deletes all temporary files used in the conversion chain.
    """
    for filename in temp_file_list:
        cmd = 'rm {}'.format(filename)
        result = sp.call(cmd.split())
        if result != 0:
            raise conversion_exceptions.FileDeletionError(
                'Unable to delete temporary file {}'.format(filename)
            )


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
            'Input file type "{}" not supported. Supported types '
            'are {}'
            ''.format(file_type, SUPPORTED_FILE_TYPES)
        )

    if file_structure not in SUPPORTED_FILE_STRUCTURES:
        raise conversion_exceptions.ImproperFileStructureError(
            'Input file structure "{}" not supported. Supported structures '
            'are {}'
            ''.format(file_structure, SUPPORTED_FILE_STRUCTURES)
        )

    if final_type not in SUPPORTED_FILE_TYPES:
        raise conversion_exceptions.ImproperFileTypeError(
            'Output file type "{}" not supported. Supported types '
            'are {}'
            ''.format(final_type, SUPPORTED_FILE_TYPES)
        )

    if final_structure not in SUPPORTED_FILE_STRUCTURES:
        raise conversion_exceptions.ImproperFileStructureError(
            'Output file structure "{}" not supported. Supported structures '
            'are {}'
            ''.format(final_structure, SUPPORTED_FILE_STRUCTURES)
        )

    if file_structure not in FILE_STRUCTURE_MAPPING[file_type]:
        raise conversion_exceptions.ImproperFileStructureError(
            'Input file structure "{structure}" is not compatible with '
            'input file type "{type}": Valid structures for {type} are '
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
            pydarnio.BorealisConvert(filename,
                                     file_type,
                                     output_file,
                                     0,     # Slice ID. TODO: Handle automatic parsing of this value from filename
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
            pydarnio.BorealisWrite(output_file,
                                   data,
                                   final_type,
                                   final_structure)
    # Downstream processing necessary
    else:
        temp_files = []     # for storing paths to temp files for later deletion

        try:
            # Convert array files to site files for processing
            if file_structure == 'array':
                reader = pydarnio.BorealisRead(filename,
                                               file_type,
                                               file_structure)
                data = reader.records
                # Generate a filename for an intermediate site file
                site_file = '{}.site'.format(file_type)
                temp_files.append(site_file)
                pydarnio.BorealisWrite(site_file,
                                       data,
                                       file_type,
                                       'site')
            else:
                site_file = filename

            # Process antennas_iq -> bfiq
            if file_type == 'antennas_iq':

                # Determine name for the bfiq file
                if final_type == 'bfiq' and final_structure == 'site':
                    bfiq_file = output_file
                else:
                    bfiq_file = 'tmp.bfiq'
                    temp_files.append(bfiq_file)

                # Convert antennas_iq.site file to bfiq.site file
                antennas_iq_to_bfiq.antennas_iq_to_bfiq(site_file, bfiq_file)

                # If bfiq is the desired output type, no more data processing necessary
                if final_type == 'bfiq':
                    # Convert to array structure if necessary
                    if final_structure == 'array':
                        reader = pydarnio.BorealisRead(bfiq_file,
                                                       'bfiq',
                                                       'site')
                        data = reader.arrays
                        pydarnio.BorealisWrite(output_file,
                                               data,
                                               'bfiq',
                                               'array')
                    remove_temp_files(temp_files)
                    return

            # For convenience
            elif file_type == 'bfiq':
                bfiq_file = site_file

            # Shouldn't ever reach this
            else:
                raise conversion_exceptions.GeneralConversionError(
                    'Unexpected error converting {} to {}'
                    ''.format(filename, output_file)
                )

            # Determine file name for rawacf.site file
            if final_structure == 'site':
                rawacf_file = output_file
            else:
                rawacf_file = 'tmp.rawacf'
                temp_files.append(rawacf_file)

            # Process bfiq -> rawacf
            bfiq_to_rawacf.bfiq_to_rawacf(bfiq_file, rawacf_file)
            # Convert to array structure
            if final_structure == 'array':
                reader = pydarnio.BorealisRead(rawacf_file,
                                               'rawacf',
                                               'site')
                data = reader.arrays
                pydarnio.BorealisWrite(output_file,
                                       data,
                                       'rawacf',
                                       'array')
            # Convert to dmap structure
            elif final_structure == 'dmap':
                pydarnio.BorealisConvert(rawacf_file,
                                         'rawacf',
                                         output_file,
                                         0,
                                         'site')
            remove_temp_files(temp_files)
            return

        # Something went wrong. Delete temporary files
        except Exception as e:
            remove_temp_files(temp_files)
            raise

