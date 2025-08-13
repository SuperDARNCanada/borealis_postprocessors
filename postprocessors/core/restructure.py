# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel

"""
This module provides some functions which are used in the processing of Borealis data files.
"""
import pydarnio
import numpy as np
import h5py


def read_group(group: h5py.Group, file_type: str, no_dim_scales=False):
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
    descriptions = {}
    units = {}
    dim_labels = {}
    dim_scales = {}
    dim_nicknames = {}

    # Get the datasets (vector fields)
    datasets = list(group.keys())
    for dset_name in datasets:
        dset = group[dset_name]
        if 'strtype' in dset.attrs.keys() or dset_name in STRING_DATASET_SIZES[file_type].keys():  # string type, requires some handling
            try:
                data = np.array([x.decode('utf-8') for x in dset[:]])
            except AttributeError:
                itemsize = dset.attrs.get('itemsize', STRING_DATASET_SIZES[file_type][dset_name])
                data = dset[:].view(dtype=(np.str_, itemsize))
        else:
            data = dset[()]  # non-string, can simply load
        group_dict[dset_name] = data

        # load in the dataset metadata for v1.0+ files
        if 'description' in dset.attrs.keys():
            descriptions[dset_name] = dset.attrs['description']
        if 'units' in dset.attrs.keys():
            units[dset_name] = dset.attrs['units']

        if not no_dim_scales:
            # Get all the information about Dimension Scales from the group
            labels = []
            nicknames = []
            scales = []
            for dim in dset.dims:
                labels.append(dim.label)  # this is the easy-to-read name, e.g. "range"
                if h5py.h5ds.is_scale(dset._id):
                    continue
                scale_nicknames = dim.keys()  # e.g. the `range_gate` field has a nickname `range gate`
                if len(scale_nicknames) == 0:
                    continue
                elif len(scale_nicknames) > 1:  # Could be multiple dim scales for a single dimension
                    nested_scales = []
                    nested_nicknames = []
                    for i, name in enumerate(scale_nicknames):
                        dim_field = dim[i]  # get the actual dataset that is the dimension scale, e.g. the `range_gate` dataset
                        scale = dim_field.name.split('/')[-1]  # get the name of that dataset, e.g. `range_gate`
                        if name == '':
                            nickname = scale
                        else:
                            nickname = name
                        dim_nicknames[scale] = nickname  # record the nickname (`range gate`) for that dimension scale dataset (`range_gate`)
                        nested_scales.append(scale)  # e.g. add `range_gate` to the list of dimension scale datasets
                        nested_nicknames.append(nickname)
                    scales.append(nested_scales)
                    nicknames.append(nested_nicknames)
                else:
                    dim_field = dim[0]
                    scale = dim_field.name.split('/')[-1]
                    scales.append(scale)
                    if scale_nicknames[0] == '':
                        nickname = scale
                    else:
                        nickname = scale_nicknames[0]
                    dim_nicknames[scale] = nickname
                    nicknames.append(nickname)
            if len(labels) > 0:
                dim_labels[dset_name] = labels
            if len(scales) > 0:
                dim_scales[dset_name] = scales  # Possibly nested list of dsets associated with each dim

        if len(descriptions) > 0:  # descriptions are required for Borealis v1.0+, so this essentially is flagging v1.0 files
            group_dict['descriptions'] = descriptions
            group_dict['units'] = units
            if not no_dim_scales:
                group_dict['dim_labels'] = dim_labels
                group_dict['dim_scales'] = dim_scales
                group_dict['dim_nicknames'] = dim_nicknames

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


def write_records(hdf5_file: h5py.File, records: dict, version=(0, 5)):
    """
    Write the record to file.

    Parameters
    ----------
    hdf5_file: h5py.File
       HDF5 file to write records to.
    records: dict
        Dictionary containing fields to write to file.
    version: tuple
        Version numbers of the record. (major, minor[, patch])
    """
    for group_name, group_dict in records.items():
        group = hdf5_file.create_group(str(group_name))

        if version[0] > 0:
            metadata = hdf5_file["metadata"]
            dim_scales = group_dict.pop("dim_scales")
            dim_labels = group_dict.pop("dim_labels")
            dim_nicknames = group_dict.pop("dim_nicknames")
            units = group_dict.pop("units")
            descriptions = group_dict.pop("descriptions")

            def write_field(name: str, is_scale=False):
                """Write a single dataset to file, in the Borealis v1.0+ format"""
                if name in metadata.keys():
                    group[name] = metadata[name]  # make a hard link to the dataset in the metadata group
                    return
                data = group_dict[name]
                field_metadata = {"description": descriptions[name]}
                if name in units.keys():
                    field_metadata["units"] = units[name]
                if name in dim_labels.keys():
                    field_metadata["dim_labels"] = dim_labels[name]
                if name in dim_scales.keys():
                    field_metadata["dim_scales"] = dim_scales[name]
                _write_hdf5_field(name, data, field_metadata, group)
                if is_scale:
                    group[name].make_scale(dim_nicknames[name])

            # determine which datasets are dimension scales for other datasets
            dim_fields = set()
            for v in dim_scales.values():
                for d in v:  # catches nested dim scales
                    if isinstance(d, list):
                        dim_fields.update(d)
                    else:
                        dim_fields.update([d])
            non_dim_fields = list(set(group_dict.keys()) - dim_fields)
            dim_fields = list(dim_fields)

            for k in dim_fields:  # Write the datasets that are dimension scales first
                write_field(k)
            for k in non_dim_fields:  # Write the datasets that are not dimension scales for other datasets last
                write_field(k)

        else:  # Borealis v0.x style
            for k, v in group_dict.items():
                if isinstance(v, str):
                    group.attrs[k] = np.bytes_(v)
                elif isinstance(v, np.ndarray):
                    if v.dtype.type == np.str_:
                        if version[1] == 5:  # version 0.5
                            dset = group.create_dataset(k, data=v.view(dtype=np.uint8))
                            dset.attrs['strtype'] = b'unicode'
                            dset.attrs['itemsize'] = v.dtype.itemsize // 4  # every character is 4 bytes
                        else:
                            group.create_dataset(k, data=np.bytes_(v))
                    else:
                        group.create_dataset(k, data=v)
                else:
                    group.attrs[k] = v


def _write_hdf5_field(
    name: str, data, metadata: dict, group: h5py.Group
):
    """
    Write ``data`` to ``group`` along with the associated ``metadata``
    """
    data = _format_for_hdf5(data)
    kw = dict()
    if not np.isscalar(data):
        kw = {"compression": "gzip", "compression_opts": 9}
    group.create_dataset(name, data=data, **kw)
    group[name].attrs["description"] = metadata.get("description")

    units = metadata.get("units", None)
    if units is not None:
        group[name].attrs["units"] = units

    dim_labels = metadata.get("dim_labels", None)
    if dim_labels is not None:
        if len(dim_labels) != len(data.shape):
            raise ValueError(
                f"{name} shape {data.shape} does not match dimension labels {dim_labels}"
            )
        for i, dim in enumerate(dim_labels):
            group[name].dims[i].label = dim

    if "dim_scales" in metadata.keys():
        _associate_dim_scales(name, group, metadata["dim_scales"])


def _associate_dim_scales(name: str, group: h5py.Group, dim_scales: list):
    """
    Associates fields as a [Dimension Scale](https://docs.h5py.org/en/stable/high/dims.html)
    of another field's dimension.
    """
    if len(group[name].shape) != len(dim_scales):
        raise ValueError(
            f"{name} has incompatible dimensionality {group[name].shape} with scales {dim_scales}"
        )
    for i, dim in enumerate(dim_scales):
        if dim is None:
            continue
        elif isinstance(dim, list):
            for d in dim:
                group[name].dims[i].attach_scale(group[d])
        else:
            group[name].dims[i].attach_scale(group[dim])


def _format_for_hdf5(field_data):
    """
    Converts ``field_data`` to supported types for a Borealis HDF5 file.
    """
    if isinstance(field_data, dict):
        return np.bytes_(str(field_data))
    elif isinstance(field_data, str):
        return np.bytes_(field_data)
    elif isinstance(field_data, bool):
        return np.bool_(field_data)
    elif isinstance(field_data, list):
        if len(field_data) > 0 and isinstance(field_data[0], str):
            return np.bytes_(field_data)
        else:
            return np.array(field_data)
    else:
        return field_data


def restructure(infile_name, outfile_name, infile_type, infile_structure, outfile_structure, version=0):
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
    version: int
        The major version of Borealis that generated infile
    """
    # dmap and iqdat are not borealis formats, so they are handled specially
    if outfile_structure == 'dmap' or outfile_structure == 'iqdat':
        pydarnio.BorealisConvert(infile_name, infile_type, outfile_name,
                                 borealis_file_structure=infile_structure)
        return

    if version == 0:
        pydarnio.BorealisRestructure(infile_name, outfile_name, infile_type, outfile_structure)
    else:
        if outfile_structure != "site":
            raise ValueError(f"Cannot restructure Borealis v1.0+ files into structure {outfile_structure}. "
                             f"Supported structures are ['site', 'dmap', 'iqdat'].")


def convert_to_numpy(data: dict, version=(0, 5)):
    """Converts lists stored in dict into numpy array. Recursive.
    Args:
        data (dict): Dictionary with lists to convert to numpy arrays.
        version (tuple): (major, minor[, patch]) version numbers
    """
    if version[0] > 0:
        return data

    for k, v in data.items():
        if isinstance(v, list):
            if len(v) > 0 and isinstance(v[0], str):
                if version[1] > 5:  # v0.6, v0.6.1, v0.7
                    data[k] = np.bytes_(v)
                else:
                    data[k] = np.array(v)
            else:
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