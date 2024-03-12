# Copyright 2021 SuperDARN Canada, University of Saskatchewan

"""
This file contains functions for converting antennas_iq files from widebeam experiments
to rawacf files, using a Hamming window in amplitude for beamforming to reduce receiver sidelobes.
Receiver beam directions are adjusted to correct for azimuthal sensitivity variations.
"""

from postprocessors.sandbox.hamming_corrected import HammingBeamformingCorrected


class Hamming8Antennas(HammingBeamformingCorrected):
    """
    Class for conversion of Borealis antennas_iq files into rawacf files for beam-broadening experiments. This class
    inherits from BaseConvert, which handles all functionality generic to postprocessing borealis files. The beams
    are formed using a Hamming window and standard beamforming, to keep the largest sidelobe down 40 dB below the
    main lobe.

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

    acf_directions = {
        10800: [-28.2, -23.46, -18.32, -13.48, -10.14, -8.2, -6.08, -2.42,
                2.42, 6.08, 8.2, 10.14, 13.48, 18.32, 23.46, 28.2],
    }
    xcf_directions = {
        10800: [-30., -24.96, -18.72, -13.18, -9.97, -8.2, -6.26, -2.62,
                2.62, 6.26, 8.2, 9.97, 13.18, 18.72, 24.96, 30.],
    }

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
