# Copyright 2022 SuperDARN Canada, University of Saskatchewan

"""
This file contains functions for extracting sequence timestamps from any file type.
"""
from collections import OrderedDict
from typing import Union

from postprocessors import BaseConvert


class ExtractBeamsFromRawacf(BaseConvert):
    """
    Class for extraction of beams from rawacf files.
    This class inherits from BaseConvert, which handles all functionality generic to postprocessing borealis files.

    See Also
    --------
    ConvertFile
    BaseConvert
    ProcessAntennasIQ2Bfiq
    ProcessBfiq2Rawacf
    ProcessAntennasIQ2Rawacf

    Attributes
    ----------
    infile: str
        The filename of the input file.
    outfile: str
        The file name of output file
    infile_structure: str
        The write structure of the file. Structures include:
        'array'
        'site'
    outfile_structure: str
        The desired structure of the output file. Same structures as above, plus 'dmap'.
    """

    def __init__(self, infile: str, outfile: str, infile_structure: str, outfile_structure: str, **kwargs):
        """
        Initialize the attributes of the class.

        Parameters
        ----------
        infile: str
            Path to input file.
        outfile: str
            Path to output file.
        infile_structure: str
            Borealis structure of input file. Either 'array' or 'site'.
        outfile_structure: str
            Borealis structure of output file. Either 'array', 'site', or 'dmap'.
        """
        super().__init__(infile, outfile, 'rawacf', 'rawacf', infile_structure, outfile_structure)

        beam_str = kwargs.get('beams', '')
        if beam_str == '':
            raise ValueError('No beams defined')

        beam_list = []
        for beam_chunk in beam_str.split(','):
            if '-' in beam_chunk:
                first_beam, last_beam = beam_chunk.split('-')
                beam_list.extend([i for i in range(int(first_beam), int(last_beam) + 1)])     # include the endpoints
            else:
                beam_list.append(int(beam_chunk))

        self.process_file(beams=beam_list)

    @staticmethod
    def process_record(record: OrderedDict, averaging_method: Union[None, str], **kwargs) -> Union[OrderedDict, None]:
        """
        Takes a record from a Borealis rawacf file and extracts only specific beams.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing rawacf data and metadata.
        averaging_method: Union[None, str]
            Method to use for averaging correlations across sequences. Acceptable methods are 'median' and 'mean'
        kwargs: dict
            beams: list of integers corresponding to beam numbers to extract.

        Returns
        -------
        record: OrderedDict
            hdf5 record, containing only the sequence timestamps.
        """
        beams = kwargs.get('beams', [])
        record_beams = record['beam_nums']

        # Find the matching beams in the record
        matching_indices = [i for i in range(len(record_beams)) if record_beams[i] in beams]

        if len(matching_indices) == 0:
            return None

        record_dims = record['correlation_dimensions']
        for field in ['main_acfs', 'intf_acfs', 'xcfs']:
            data = record[field]
            data = data.reshape(record_dims)
            # data is [num_beams, num_ranges, num_lags]
            record[field] = data[matching_indices].flatten()

        record_dims[0] = len(matching_indices)

        record['correlation_dimensions'] = record_dims
        record['beam_azms'] = record['beam_azms'][matching_indices]
        record['beam_nums'] = record['beam_nums'][matching_indices]

        return record
