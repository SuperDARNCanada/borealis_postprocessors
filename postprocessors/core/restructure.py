# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel

"""
This module provides some functions which are used in the processing of Borealis data files.
"""
import pydarnio
import numpy as np
import h5py


def read_group(group: h5py.Group):
    """
    Reads a group from an HDF5 file into a dictionary.

    Parameters
    ----------
    group: h5py.Group
        Opened h5py group

    Returns
    -------
    dict
        Dictionary of {group_name: {}} where the inner dictionary is the datasets/attributes
        of the hdf5 group.
    """
    group_dict = {}
    # Get the datasets (vector fields)
    datasets = list(group.keys())
    for dset_name in datasets:
        dset = group[dset_name]
        if 'strtype' in dset.attrs:  # string type, requires some handling
            itemsize = dset.attrs['itemsize']
            data = dset[:].view(dtype=(np.unicode_, itemsize))
        else:
            data = dset[:]  # non-string, can simply load
        group_dict[dset_name] = data

    # Get the attributes (scalar fields)
    attribute_dict = {}
    for k, v in group.attrs.items():
        if k in ['CLASS', 'TITLE', 'VERSION', 'DEEPDISH_IO_VERSION', 'PYTABLES_FORMAT_VERSION']:
            continue
        elif isinstance(v, np.bytes_):
            attribute_dict[k] = v.tobytes().decode('utf-8')
        elif isinstance(v, h5py.Empty):
            dtype = v.dtype.type
            data = dtype()
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            attribute_dict[k] = data
        else:
            attribute_dict[k] = v
    group_dict.update(attribute_dict)

    return group_dict


def write_records(hdf5_file: h5py.File, records: dict):
    """
    Write the record to file.

    Parameters
    ----------
    hdf5_file: h5py.File
       HDF5 file to write records to.
    records: dict
        Dictionary containing fields to write to file.
    """
    for group_name, group_dict in records.items():
        group = hdf5_file.create_group(str(group_name))
        for k, v in group_dict.items():
            array_field = False
            if isinstance(v, str):
                data = np.bytes_(v)
            elif isinstance(v, bool):
                data = np.bool_(v)
            elif isinstance(v, list):
                if isinstance(v[0], str):
                    data = np.bytes_(v)
                else:
                    data = np.array(v)
                array_field = True
            elif isinstance(v, np.ndarray):
                if isinstance(v[0], str):
                    data = np.bytes_(v)
                else:
                    data = v
                array_field = True
            else:
                data = v

            # Store in the file appropriately
            if array_field:
                group.create_dataset(k, data=data)
            else:
                group.attrs[k] = data


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
