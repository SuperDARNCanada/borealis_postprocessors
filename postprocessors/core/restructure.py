# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel

"""
This module provides some functions which are used in the processing of Borealis data files.
"""
import pydarnio
import numpy as np
import h5py


def read_group(group: h5py.Group, file_type: str):
    """
    Reads a group from an HDF5 file into a dictionary.

    Parameters
    ----------
    group: h5py.Group
        Opened h5py group
    file_type: str
        Type of data file. One of 'antennas_iq', 'bfiq', or 'rawacf'

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
        if 'strtype' in dset.attrs.keys() or dset_name in STRING_DATASET_SIZES[file_type].keys():  # string type, requires some handling
            data = np.array([x.decode('utf-8') for x in dset[:]])
        else:
            data = dset[:]  # non-string, can simply load
        group_dict[dset_name] = data

    # Get the attributes (scalar fields)
    attribute_dict = {}
    for k, v in group.attrs.items():
        if k in ['CLASS', 'TITLE', 'VERSION', 'DEEPDISH_IO_VERSION', 'PYTABLES_FORMAT_VERSION']:
            continue
        elif isinstance(v, np.bytes_):
            if v.itemsize == 0:
                attribute_dict[k] = ''
            else:
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
            if isinstance(v, str):
                group.attrs[k] = np.bytes_(v)
            elif isinstance(v, np.ndarray):
                if v.dtype.type == np.str_:
                    dset = group.create_dataset(k, data=v.view(dtype=np.uint8))
                    dset.attrs['strtype'] = b'unicode'
                    dset.attrs['itemsize'] = v.dtype.itemsize // 4  # every character is 4 bytes
                else:
                    group.create_dataset(k, data=v)
            else:
                group.attrs[k] = v


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

# Maps the string-type dataset fields of a record to their corresponding numpy-array itemsizes, for correct parsing
# of these fields. This dictionary contains the fallback values if the dataset itself doesn't contain the metadata
# that is expected for these fields.
STRING_DATASET_SIZES = {
    'antennas_iq': {'antenna_arrays_order': 10,
                    'data_descriptors': 13},
    'bfiq': {'antenna_arrays_order': 4,
             'data_descriptors': 18},
    'rawacf': {'correlation_descriptors': 10,
               'data_descriptors': 10}
}