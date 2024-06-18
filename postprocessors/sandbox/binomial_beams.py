# Copyright 2023 SuperDARN Canada, University of Saskatchewan

"""
This file contains functions for converting antennas_iq files from widebeam experiments
to rawacf files, using a Binomial window in amplitude for beamforming to reduce receiver sidelobes.
"""
from postprocessors import AntennasIQ2Rawacf


class BinomialWindowBeamforming(AntennasIQ2Rawacf):
    """
    Class for conversion of Borealis antennas_iq files into rawacf files for beam-broadening experiments. This class
    inherits from BaseConvert, which handles all functionality generic to postprocessing borealis files. The beams
    are formed using a Binomial window and standard beamforming, to remove sidelobes.

    See Also
    --------
    ConvertFile
    BaseConvert
    HammingWindowBeamforming

    Attributes
    ----------
    infile: str
        The filename of the input antennas_iq file.
    outfile: str
        The file name of output file
    infile_structure: str
        The write structure of the file. Structures include:
        'array'
        'site'
    outfile_structure: str
        The desired structure of the output file. Same structures as above, plus 'dmap'.
    """
    window = [0.0001554001554001554, 0.002331002331002331, 0.016317016317016316, 0.0707070707070707,
              0.21212121212121213, 0.4666666666666667, 0.7777777777777778, 1.0,
              1.0, 0.7777777777777778, 0.4666666666666667, 0.21212121212121213,
              0.0707070707070707, 0.016317016317016316, 0.002331002331002331, 0.0001554001554001554]

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
        super().__init__(infile, outfile, infile_structure, outfile_structure)

        self.process_file(**kwargs)
