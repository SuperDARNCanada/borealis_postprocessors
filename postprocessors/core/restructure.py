# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel

"""
This module provides some functions which are used in the processing of Borealis data files.
"""
import pydarnio
import numpy as np


def restructure(infile_name, outfile_name, infile_type, infile_structure, outfile_structure):
    """
    This method restructures filename of structure "file_structure" into "final_structure".

    Parameters
    ----------
    infile_name: str
        Name of the original file.
    outfile_name: str
        Name of the restructured file.
    infile_type: str
        Borealis file type of the files.
    infile_structure: str
        The current write structure of the file. One of 'array' or 'site'.
    outfile_structure: str
        The desired write structure of the file. One of 'array', 'site', 'iqdat', or 'dmap'.
    """
    # dmap and iqdat are not borealis formats, so they are handled specially
    if outfile_structure == 'dmap' or outfile_structure == 'iqdat':
        pydarnio.BorealisConvert(infile_name, infile_type, outfile_name,
                                 borealis_file_structure=infile_structure)
        return

    pydarnio.BorealisRestructure(infile_name, outfile_name, infile_type, outfile_structure)


def convert_to_numpy(data: dict):
    """Converts lists stored in dict into numpy array. Recursive.
    Args:
        data (Python dictionary): Dictionary with lists to convert to numpy arrays.
    """
    for k, v in data.items():
        if isinstance(v, dict):
            convert_to_numpy(v)
        elif isinstance(v, list):
            data[k] = np.array(v)
        else:
            continue

    return data


# Dictionary mapping the accepted borealis file types to the borealis file types that they can be processed into.
# The dictionary keys are the valid input file types, and their values are lists of file types which they can be
# processed into.
FILE_TYPE_MAPPING = {
    'antennas_iq': ['antennas_iq', 'bfiq', 'rawacf'],
    'bfiq': ['bfiq', 'rawacf'],
    'rawacf': ['rawacf']
}

# Dictionary mapping the accepted borealis file types to the borealis file structures that they can be formatted as.
# The dictionary keys are the valid input file types, and their values are lists of file structures which they can be
# formatted as.
FILE_STRUCTURE_MAPPING = {
    'antennas_iq': ['site', 'array'],
    'bfiq': ['site', 'array', 'iqdat'],
    'rawacf': ['site', 'array', 'dmap']
}
