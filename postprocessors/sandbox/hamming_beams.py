# Copyright 2021 SuperDARN Canada, University of Saskatchewan

"""
This file contains functions for converting antennas_iq files from widebeam experiments
to rawacf files, using a Hamming window in amplitude for beamforming to reduce receiver sidelobes.
"""
from typing import Union

from postprocessors import AntennasIQ2Rawacf


class HammingWindowBeamforming(AntennasIQ2Rawacf):
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
    window = [0.08081232549588463, 0.12098514265395757, 0.23455777475180511, 0.4018918165398586,
              0.594054435182454, 0.7778186328978896, 0.9214100134552521, 1.0,
              1.0, 0.9214100134552521, 0.7778186328978896, 0.594054435182454,
              0.4018918165398586, 0.23455777475180511, 0.12098514265395757, 0.08081232549588463]

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
        beam_azms: list[float]
            List of all beam directions (in degrees) to reprocess into
        beam_nums: list[uint]
            List describing beam order. Numbers in this list correspond to indices of beam_azms
        """
        super().__init__(infile, outfile, infile_structure, outfile_structure)

        self.process_file(**kwargs)

