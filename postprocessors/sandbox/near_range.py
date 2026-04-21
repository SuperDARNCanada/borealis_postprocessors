"""
This file contains functions for converting antennas_iq files
to rawacf files, starting from a near range of 90km.
"""
from collections import OrderedDict
import h5py
import numpy as np

from postprocessors import AntennasIQ2Rawacf


class NearRange(AntennasIQ2Rawacf):
    """
    Class for conversion of Borealis antennas_iq files into rawacf files. This class inherits from
    BaseConvert, which handles all functionality generic to postprocessing borealis files.

    Attributes
    ----------
    infile: str
        The filename of the input antennas_iq file.
    outfile: str
        The file name of output file
    infile_structure: str
        The structure of the file. Structures include:
        'array'
        'site'
    outfile_structure: str
        The desired structure of the output file. Same structures as above, plus 'dmap'.
    """

    def __init__(self, infile: str, outfile: str, infile_structure: str, outfile_structure: str):
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
        super().__init__(infile, outfile, infile_structure, outfile_structure)

    def process_file(self, **kwargs):
        super().process_file(first_range=45.0, num_ranges=78, **kwargs)

    @classmethod
    def _update_metadata(cls, record: OrderedDict, metadata: h5py.Group, **kwargs):
        description = metadata["range_gates"].attrs["description"]
        del metadata["range_gates"]
        dset = metadata.create_dataset("range_gates", data=np.arange(78))
        dset.attrs["description"] = description
        dset.make_scale()

        first_range = 45.0
        metadata["first_range"][()] = first_range
        metadata["first_range_rtt"][()] = np.float32(first_range * 2.0 * 1.0e3 * 1.0e6 / 299_792_458)
        super()._update_metadata(record, metadata, **kwargs)

