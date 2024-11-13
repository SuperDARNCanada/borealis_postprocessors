# Copyright 2021 SuperDARN Canada, University of Saskatchewan

"""
This file contains functions for converting antennas_iq files from widebeam experiments
to rawacf files, using a Chebyshev window in amplitude for beamforming to reduce receiver sidelobes.
"""
from collections import OrderedDict

import numpy as np
from postprocessors import AntennasIQ2Rawacf


class Chebyshev30dB(AntennasIQ2Rawacf):
    """
    Class for conversion of Borealis antennas_iq files into rawacf files for beam-broadening experiments. This class
    inherits from BaseConvert, which handles all functionality generic to postprocessing borealis files. The beams
    are formed using a Hamming window and standard beamforming, to keep the sidelobes down 30 dB below the
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
    window = [0.2910, 0.3173, 0.4557, 0.6018, 0.7424, 0.8637, 0.9528, 1.0000,
              1.0000, 0.9528, 0.8637, 0.7424, 0.6018, 0.4557, 0.3173, 0.2910]

    xcf_directions = {
        10400: [-24.9, -20.66, -17.52, -14.88, -10.94, -7.7, -5.23, -2.12,
                2.12, 5.23, 7.7, 10.94, 14.88, 17.52, 20.66, 24.9],
        10500: [-25.2, -20.86, -17.56, -14.88, -10.94, -7.6, -5.28, -2.02,
                2.02, 5.28, 7.6, 10.94, 14.88, 17.56, 20.86, 25.2],
        10600: [-24.5, -20.56, -17.92, -14.48, -10.94, -8.1, -4.68, -1.72,
                1.72, 4.68, 8.1, 10.94, 14.48, 17.92, 20.56, 24.5],
        10700: [-24.6, -20.46, -17.62, -14.68, -11.22, -8.2, -4.96, -1.82,
                1.82, 4.96, 8.2, 11.22, 14.68, 17.62, 20.46, 24.6],
        10800: [-24.5, -20.46, -17.82, -14.49, -10.74, -8.05, -5.16, -2.02,
                2.02, 5.16, 8.05, 10.74, 14.49, 17.82, 20.46, 24.5],
        10900: [-24.7, -20.56, -17.76, -14.58, -10.64, -7.9, -5.16, -2.22,
                2.22, 5.16, 7.9, 10.64, 14.58, 17.76, 20.56, 24.7],
        12200: [-24.25, -21.16, -17.51, -14.68, -10.64, -8.1, -4.76, -2.12,
                2.12, 4.76, 8.1, 10.64, 14.68, 17.51, 21.16, 24.25],
        12300: [-24.05, -21.46, -17.62, -14.58, -10.64, -8.0, -4.93, -2.21,
                2.21, 4.93, 8.0, 10.64, 14.58, 17.62, 21.46, 24.05],
        12500: [-23.7, -21.23, -17.92, -14.88, -10.64, -8.0, -4.86, -2.22,
                2.22, 4.86, 8.0, 10.64, 14.88, 17.92, 21.23, 23.7],
        13000: [-23.8, -21.38, -18.22, -14.48, -10.54, -8.1, -4.66, -2.22,
                2.22, 4.66, 8.1, 10.54, 14.58, 18.22, 21.38, 23.8],
        13100: [-23.7, -21.18, -18.22, -13.68, -10.77, -8.5, -4.86, -2.42,
                2.42, 4.86, 8.5, 10.77, 13.68, 18.22, 21.18, 23.7],
        13200: [-24.4, -21.06, -18.22, -14.58, -10.94, -7.5, -4.63, -2.42,
                2.42, 4.63, 7.5, 10.94, 14.58, 18.22, 21.06, 24.4],
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

    @classmethod
    def process_record(cls, record: OrderedDict, **kwargs) -> OrderedDict:
        """
        Overwrites the beam directions before beamforming then passes off to parent class processing.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing antennas_iq data and metadata

        Returns
        -------
        record: OrderedDict
            hdf5 record, with new fields required by bfiq data format
        """
        beam_nums = record['beam_nums']
        freq_khz = record['freq']
        old_beam_azms = record['beam_azms']
        new_beam_azms = np.array([cls.xcf_directions[freq_khz][i] for i in beam_nums])
        record['beam_azms'] = new_beam_azms

        # Now do the processing with the new beam directions
        record = super().process_record(record)

        # The direction of sensitivity is still the same, so revert back to original beam directions.
        record['beam_azms'] = old_beam_azms

        return record
