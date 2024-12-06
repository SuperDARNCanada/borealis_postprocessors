# Copyright 2021 SuperDARN Canada, University of Saskatchewan

"""
This file contains functions for converting antennas_iq files from widebeam experiments
to rawacf files, using a Chebyshev window in amplitude for beamforming to reduce receiver sidelobes.
"""
from collections import OrderedDict
import copy

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
        10400: [-26.5, -21.8, -18.5, -15.6, -11.5, -8.1, -5.5, -2.2,
                2.2, 5.5, 8.1, 11.5, 15.6, 18.5, 21.8, 26.5],
        10500: [-26.8, -22., -18.5, -15.6, -11.5, -8.1, -5.5, -2.2,
                2.2, 5.5, 8.1, 11.5, 15.6, 18.5, 22., 26.8],
        10600: [-26., -21.7, -18.9, -15.2, -11.5, -8.6, -5., -1.8,
                1.8, 5., 8.6, 11.5, 15.2, 18.9, 21.7, 26.],
        10700: [-26., -21.5, -18.8, -15.2, -11.3, -8.5, -5.4, -2.2,
                2.2, 5.4, 8.5, 11.3, 15.2, 18.8, 21.5, 26.],
        10800: [-26., -21.5, -18.8, -15.2, -11.3, -8.5, -5.4, -2.2,
                2.2, 5.4, 8.5, 11.3, 15.2, 18.8, 21.5, 26.],
        10900: [-26.2, -21.7, -18.7, -15.3, -11.2, -8.3, -5.4, -2.2,
                2.2, 5.4, 8.3, 11.2, 15.3, 18.7, 21.7, 26.2],
        12200: [-25.4, -22.1, -18.2, -15.2, -11.1, -8.4, -4.9, -2.3,
                2.3, 4.9, 8.4, 11.1, 15.2, 18.2, 22.1, 25.4],
        12300: [-25.1, -22.3, -18.3, -15.2, -11.1, -8.3, -5., -2.2,
                2.2, 5., 8.3, 11.1, 15.2, 18.3, 22.3, 25.1],
        12500: [-24.7, -22.1, -18.5, -15.5, -11., -8.3, -5., -2.3,
                2.3, 5., 8.3, 11., 15.5, 18.5, 22.1, 24.7],
        13000: [-24.8, -22.2, -18.8, -15., -10.9, -8.4, -4.8, -2.2,
                2.2, 4.8, 8.4, 10.9, 15., 18.8, 22.2, 24.8],
        13100: [-24.7, -22., -18.8, -14.1, -11.2, -8.8, -5., -2.5,
                2.5, 5., 8.8, 11.2, 14.1, 18.8, 22., 24.7],
        13200: [-25.3, -21.8, -18.8, -15.1, -11.3, -7.7, -4.8, -2.5,
                2.5, 4.8, 7.7, 11.3, 15.1, 18.8, 21.8, 25.3],
    }
    acf_directions = {
        10400: [-25.7, -21.4, -18.2, -15.4, -11.3, -8., -5.3, -2.1,
                2.1, 5.3, 8., 11.3, 15.4, 18.2, 21.4, 25.7],
        10500: [-25.8, -21.5, -18.1, -15.4, -11.3, -8., -5.3, -2.1,
                2.1, 5.3, 8., 11.3, 15.4, 18.1, 21.5, 25.8],
        10600: [-25.3, -21.2, -18.6, -14.9, -11.3, -8.5, -4.8, -1.8,
                1.8, 4.8, 8.5, 11.3, 14.9, 18.6, 21.2, 25.3],
        10700: [-26., -21.5, -18.8, -15.2, -11.3, -8.5, -5.4, -2.2,
                2.2, 5.4, 8.5, 11.3, 15.2, 18.8, 21.5, 26.],
        10800: [-25.2, -21.2, -18.5, -15., -11.2, -8.4, -5.2, -2.1,
                2.1, 5.2, 8.4, 11.2, 15., 18.5, 21.2, 25.2],
        10900: [-25.4, -21.3, -18.4, -15.1, -11.1, -8.2, -5.2, -2.1,
                2.1, 5.2, 8.2, 11.1, 15.1, 18.4, 21.3, 25.4],
        12200: [-24.8, -21.8, -17.9, -15.1, -11., -8.5, -4.8, -2.2,
                2.2, 4.8, 8.5, 11., 15.1, 17.9, 21.8, 24.8],
        12300: [-24.6, -22., -17.9, -15.1, -11., -8.3, -4.9, -2.2,
                2.2, 4.9, 8.3, 11., 15.1, 17.9, 22., 24.6],
        12500: [-24.3, -21.8, -18.1, -15.3, -11., -8.4, -4.8, -2.2,
                2.2, 4.8, 8.4, 11., 15.3, 18.1, 21.8, 24.3],
        13000: [-24.3, -21.8, -18.4, -14.9, -10.9, -8.5, -4.6, -2.2,
                2.2, 4.6, 8.5, 10.9, 14.9, 18.4, 21.8,  24.3],
        13100: [-24.2, -21.7, -18.5, -14.2, -11.2, -8.7, -4.7, -2.4,
                2.4, 4.7, 8.7, 11.1, 14.2, 18.6, 21.7, 24.2],
        13200: [-24.7, -21.5, -18.5, -14.9, -11.3, -7.8, -4.7, -2.4,
                2.4, 4.7, 7.8, 11.3, 14.9, 18.5, 21.5, 24.7],
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
        acf_record = copy.deepcopy(record)
        beam_nums = record['beam_nums']
        freq_khz = record['freq']
        old_beam_azms = record['beam_azms']
        acf_beam_azms = np.array([cls.acf_directions[freq_khz][i] for i in beam_nums])
        xcf_beam_azms = np.array([cls.xcf_directions[freq_khz][i] for i in beam_nums])
        acf_record['beam_azms'] = acf_beam_azms
        record['beam_azms'] = xcf_beam_azms

        # Now do the processing with the new beam directions
        acf_record = super().process_record(acf_record)
        record = super().process_record(record)

        record['main_acfs'] = acf_record['main_acfs']

        # The direction of sensitivity is still the same, so revert back to original beam directions.
        record['beam_azms'] = old_beam_azms

        return record