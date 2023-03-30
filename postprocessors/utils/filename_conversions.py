"""
Copyright SuperDARN Canada 2022

Functions for converting a Borealis filename into a new filename for a downstream or
restructured file.
"""


def borealis_to_borealis_rename(filename, new_type, new_structure):
    """
    Returns a typical Borealis-formatted filename with the new type and structure.
    Typical Borealis-formatted filenames are YYYYMMDD.HHMM.SS.radar_id.slice_num.type.hdf5[.site]
    where
    radar id = three letter code, e.g. 'sas'
    slice_num = number, e.g. 0
    type = borealis data type, one of 'antennas_iq', 'bfiq', or 'rawacf'
    .site = optional file ending

    Parameters
    ----------
    filename: str
        Name of original file
    new_type: str
        New Borealis data type. 'antennas_iq', 'bfiq', or 'rawacf'.
    new_structure: str
        New Borealis file structure. 'site' or 'array'

    Returns
    -------
    new_name: str
        Newly formatted Borealis filename
    """
    fields = filename.split(".")
    if new_structure == 'site' and fields[-1] != 'site':
        fields.append('site')
    elif new_structure == 'array' and fields[-1] != 'hdf5':
        fields.pop(-1)

    new_name = '.'.join(fields[0:5] + [new_type] + fields[6:])

    return new_name


def borealis_to_sdarn_rename(filename, new_type):
    """
    Returns a typical SuperDARN-formatted filename with the new type and structure.
    Typical Borealis-formatted filenames are YYYYMMDD.HHMM.SS.radar_id.slice_num.type.hdf5[.site]
    Typical SuperDARN-formatted filenames are YYYYMMDD.HHMM.SS.radar_id.alphabetic_slice_id.type
    where
    radar id = three letter code, e.g. 'sas'
    slice_num = number, e.g. 0
    alphabetic_slice_id = single letter id, so if slice_num = 0, this = 'a', 1->'b', 2->'c', etc.
    type = data type, one of 'antennas_iq', 'bfiq', or 'rawacf' for Borealis files,
           one of 'iqdat', 'rawacf' for SuperDARN-formatted files.
    .site = optional file ending

    Parameters
    ----------
    filename: str
        Name of original file
    new_type: str
        New Borealis data type. 'iqdat' or 'rawacf'

    Returns
    -------
    new_name: str
        Newly formatted Borealis filename
    """
    fields = filename.split('.')
    slice_id = fields[4]
    slice_char = chr(int(slice_id) + ord('a'))
    sdarn_file = '.'.join(fields[0:4] + [slice_char, new_type])
    return sdarn_file
