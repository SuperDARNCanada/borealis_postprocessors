# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains functions for opening and converting
SuperDARN data files.
"""


import os


FILE_TYPE_MAPPING = {
    'antennas_iq': ['antennas_iq', 'bfiq', 'rawacf'],
    'bfiq': ['bfiq', 'rawacf'],
    'rawacf': ['rawacf']
}

FILE_STRUCTURE_MAPPING = {
    'antennas_iq': ['site', 'array'],
    'bfiq': ['site', 'array'],
    'rawacf': ['site', 'array', 'dmap']
}


def convert_file(filename: str, file_type: str, final_type: str,
                 file_structure: str = 'array', final_structure: str = 'array'):
    """
    Reads a SuperDARN data file, and converts it to the desired file
    type and structure.

    Parameters
    ----------
        filename: str
            file name containing SuperDARN data.
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
    if file_type not in ['antennas_iq', 'bfiq', 'rawacf']:
        # TODO: Raise exception here
        return

    if file_structure not in ['array', 'site', 'dmap']:
        # TODO: Raise exception here
        return

    if final_type not in ['antennas_iq', 'bfiq', 'rawacf']:
        # TODO: Raise exception here
        return

    if final_structure not in ['array', 'site', 'dmap']:
        # TODO: Raise exception here
        return

    if file_structure not in FILE_STRUCTURE_MAPPING[file_type]:
        # TODO: Raise exception here
        return
    
    if final_structure not in FILE_TYPE_MAPPING[final_type]:
        # TODO: Raise exception here
        return

    if not os.path.isfile(filename):
        # TODO: Raise exception here
        return
