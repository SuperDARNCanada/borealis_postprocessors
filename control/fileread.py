# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains functions for opening and converting
SuperDARN data files.
"""


def borealis_convert(filename: str, filetype: str, final_type: str,
                    file_structure: str = 'array',
                    final_file_structure: str = 'array'):
    """
    Reads a SuperDARN data file, and converts it to the desired file
    type and structure.

    Parameters
    ----------
        filename: str
            file name containing SuperDARN data.
        filetype: str
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
        final_file_structure:
            The desired structure of the output file. Same structures as
            above.
    """
    pass
