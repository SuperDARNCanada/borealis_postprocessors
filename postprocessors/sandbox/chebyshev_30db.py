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

    ### The directions below are for a 60 degree FOV, with element factor considered in the weights ###
    widebeam_60deg_xcf_directions = {
        10400: [-25.5, -21.6, -18.7, -15.8, -11.5, -7.7, -5.1, -2.1,
                2.1, 5.1, 7.7, 11.5, 15.8, 18.7, 21.6, 25.5],
        10500: [-25.2, -20.7, -17.9, -15., -12.1, -8.8, -5., -1.9,
                1.9, 5., 8.8, 12.1, 15., 17.9, 20.7, 25.2],
        10600: [-25.5, -20.7, -17.8, -15., -12.1, -8.7, -4.9, -1.9,
                1.9, 4.9, 8.7, 12.1, 15., 17.8, 20.7, 25.5],
        10700: [-24.9, -21.6, -18.4, -15.6, -11.5, -7.7, -5.2, -2.1,
                2.1, 5.2, 7.7, 11.5, 15.6, 18.4, 21.6, 24.9],
        10800: [-25.5, -21., -17.9, -15.5, -11.7, -7.7, -4.8, -2.,
                2., 4.8, 7.7, 11.7, 15.5, 17.9, 21., 25.5],
        10900: [-25.5, -21., -17.8, -15.4, -11.8, -7.7, -4.7, -2.,
                2., 4.7, 7.7, 11.8, 15.4, 17.8, 21., 25.5],
        12200: [-24.7, -21.6, -17.7, -14.2, -11.5, -8.2, -4.9, -1.8,
                1.8, 4.9, 8.2, 11.5, 14.2, 17.7, 21.6, 24.7],
        12300: [-24.6, -21.6, -17.5, -14.4, -11.4, -7.9, -5.1, -2.1,
                2.1, 5.1, 7.9, 11.4, 14.4, 17.5, 21.6, 24.6],
        12500: [-24.6, -21.5, -17.7, -13.9, -11.3, -8.3, -4.9, -1.8,
                1.8, 4.9, 8.3, 11.3, 13.9, 17.7, 21.5, 24.6],
        13000: [-24.2, -21.7, -18.6, -14.9, -11.4, -7.6, -4.6, -2.4,
                2.4, 4.6, 7.6, 11.4, 14.9, 18.6, 21.7, 24.2],
        13100: [-24.9, -21.2, -18.4, -13.5, -11., -8.5, -4.7, -1.5,
                1.5, 4.7, 8.5, 11., 13.5, 18.4, 21.2, 24.9],
        13200: [-25.1, -22., -18.6, -14., -11.7, -8.5, -4.8, -2.3,
                2.3, 4.8, 8.5, 11.7, 14., 18.6, 22., 25.1],
    }
    widebeam_60deg_acf_directions = {
        10400: [-25., -21.2, -18.3, -15.5, -11.4, -7.7, -5., -2.1,
                2.1, 5., 7.7, 11.4, 15.5, 18.3, 21.2, 25.],
        10500: [-24.8, -20.7, -17.9, -14.8, -11.8, -8.6, -4.9, -1.9,
                1.9, 4.9, 8.6, 11.8, 14.8, 17.9, 20.7, 24.8],
        10600: [-25., -20.7, -17.8, -14.9, -11.9, -8.5, -4.8, -1.9,
                1.9, 4.8, 8.5, 11.9, 14.9, 17.8, 20.7, 25.],
        10700: [-24.5, -21.4, -18.1, -15.3, -11.5, -7.7, -5.1, -2.1,
                2.1, 5.1, 7.7, 11.5, 15.3, 18.1, 21.4, 24.5],
        10800: [-25., -20.9, -17.8, -15.3, -11.6, -7.8, -4.8, -2.1,
                2.1, 4.8, 7.8, 11.6, 15.3, 17.8, 20.9, 25.],
        10900: [-24.9, -20.9, -17.7, -15.3, -11.7, -7.8, -4.7, -2.,
                2., 4.7, 7.8, 11.7, 15.3, 17.7, 20.9, 25.],
        12200: [-24.4, -21.5, -17.7, -14.2, -11.5, -8.2, -4.8, -1.8,
                1.8, 4.8, 8.2, 11.5, 14.2, 17.7, 21.5, 24.4],
        12300: [-24.2, -21.5, -17.5, -14.4, -11.5, -7.9, -5., -2.1,
                2.1, 5., 7.9, 11.5, 14.4, 17.5, 21.5, 24.2],
        12500: [-24.3, -21.4, -17.8, -14.1, -11.4, -8.2, -4.9, -1.8,
                1.8, 4.9, 8.2, 11.4, 14.1, 17.8, 21.4, 24.3],
        13000: [-23.9, -21.5, -18.4, -14.8, -11.4, -7.8, -4.6, -2.4,
                2.4, 4.6, 7.8, 11.4, 14.8, 18.4, 21.5, 23.9],
        13100: [-24.5, -21., -18.4, -13.7, -11.1, -8.5, -4.7, -1.5,
                1.5, 4.7, 8.5, 11.1, 13.7, 18.4, 21., 24.5],
        13200: [-24.8, -21.6, -18.4, -14., -11.7, -8.4, -4.6, -2.2,
                2.2, 4.6, 8.4, 11.7, 14., 18.4, 21.6, 24.8],
    }

    ### The directions below are for a 60-degree FOV, no element factor applied in generation ###
    widebeam_60deg_droop_xcf_directions = {
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
    widebeam_60deg_droop_acf_directions = {
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

    ### The directions below are for a nominal FOV, without element factor considered in the weights ###
    original_widebeam_xcf_directions = {
        10500: [-27.1, -22.7, -19.3, -14.5, -11.3, -7.7, -3.7, -1.4,
                1.4, 3.7, 7.7, 11.3, 14.5, 19.3, 22.7, 27.1],
        10600: [-27.2, -22.7, -19.3, -14.6, -11.3, -7.8, -3.7, -1.4,
                1.4, 3.7, 7.8, 11.3, 14.6, 19.3, 22.7, 27.2],
        10700: [-27.4, -22.6, -19.4, -14.6, -11.3, -7.8, -3.7, -1.4,
                1.4, 3.7, 7.8, 11.3, 14.6, 19.4, 22.6, 27.4],
        10800: [-27.5, -22.6, -19.4, -14.7, -11.3, -7.9, -3.7, -1.4,
                1.4, 3.7, 7.9, 11.2, 14.7, 19.4, 22.6, 27.5],
        10900: [-27.6, -22.5, -19.4, -14.8, -11.2, -7.9, -3.7, -1.4,
                1.4, 3.7, 7.9, 11.2, 14.8, 19.4, 22.5, 27.6],
        12200: [-27., -21.7, -17.4, -14.8, -11.2, -8.1, -4.9, -1.8,
                1.8, 4.9, 8.1, 11.2, 14.8, 17.4, 21.7, 27.],
        12300: [-27.1, -21.8, -17.5, -14.8, -11.2, -8.1, -4.9, -1.8,
                1.8, 4.9, 8.1, 11.2, 14.8, 17.5, 21.8, 27.1],
        12500: [-27.3, -21.9, -17.5, -14.7, -11.3, -7.9, -4.9, -1.9,
                1.9, 4.9, 7.9, 11.3, 14.7, 17.5, 21.9, 27.3],
        13000: [-27.7, -22.1, -17.5, -14.6, -11.2, -7.6, -5., -2.2,
                2.2, 5., 7.6, 11.2, 14.6, 17.5, 22.1, 27.7],
        13100: [-28.2, -22.2, -17.6, -14.5, -11.4, -7.6, -4.9, -2.2,
                2.2, 4.9, 7.6, 11.4, 14.5, 17.6, 22.2, 28.2],
        13200: [-28.1, -22.3, -17.7, -14.5, -11.3, -7.6, -5., -2.2,
                2.2, 5., 7.6, 11.3, 14.5, 17.7, 22.3, 28.1],
    }
    original_widebeam_acf_directions = {
        10500: [-26., -21.9, -18.9, -14.3, -11.4, -8., -3.9, -1.6,
                1.6, 3.9, 8., 11.4, 14.3, 18.9, 21.9, 26.],
        10600: [-26.1, -21.9, -18.9, -14.4, -11.3, -8., -3.9, -1.6,
                1.6, 3.9, 8., 11.3, 4.4, 18.9, 21.9, 26.1],
        10700: [-26.2, -21.8, -18.9, -14.4, -11.3, -8.1, -3.9, -1.6,
                1.6, 3.9, 8., 11.3, 14.4, 18.9, 21.8, 26.2],
        10800: [-26.3, -21.8, -18.9, -14.5, -11.3, -8.1, -3.9, -1.6,
                1.6, 3.9, 8.1, 11.3, 14.5, 18.9, 21.8, 26.3],
        10900: [-26.4, -21.7, -18.9, -14.6, -11.3, -8.1, -3.9, -1.6,
                1.6, 3.9, 8.1, 11.3, 14.6, 18.9, 21.7, 26.4],
        12200: [-25.8, -21.4, -17.4, -14.9, -11.2, -8.1, -4.9, -1.8,
                1.8, 4.9, 8.1, 11.2, 14.9, 17.4, 21.4, 25.8],
        12300: [-25.8, -21.5, -17.4, -14.8, -11.2, -8.1, -4.9, -1.8,
                1.8, 4.9, 8.1, 11.2, 14.8, 17.4, 21.5, 25.8],
        12500: [-25.9, -21.6, -17.4, -14.8, -11.3, -8., -4.9, -1.9,
                1.9, 4.9, 7.9, 11.3, 14.8, 17.4, 21.6, 25.9],
        13000: [-26.2, -21.7, -17.5, -14.7, -11.3, -7.7, -4.9, -2.1,
                2.1, 4.9, 7.7, 11.3, 14.7, 17.5, 21.7, 26.2],
        13100: [-26.5, -21.7, -17.6, -14.5, -11.5, -7.7, -4.9, -2.1,
                2.1, 4.9, 7.7, 11.5, 14.5, 17.6, 21.7, 26.5],
        13200: [-26.5, -21.7, -17.6, -14.6, -11.4, -7.7, -4.9, -2.1,
                2.1, 4.9, 7.7, 11.4, 14.6, 17.6, 21.7, 26.5],
    }

    ### The directions below are for a nominal FOV with 8 TX antennas, without element factor considered in the weights ###
    original_8tx_widebeam_xcf_directions = {
        10600: [-28.3, -23.6, -18.2, -13.4, -10.2, -8.3, -6.2, -2.5,
                2.5, 6.2, 8.3, 10.2, 13.4, 18.2, 23.6, 28.3],
        13100: [-26., -23.1, -19.5, -14.8, -10., -7., -5.3, -2.4,
                2.4, 5.3, 7., 10., 14.8, 19.5, 23.1, 26.],
    }
    original_8tx_widebeam_acf_directions = {
        10600: [-26.9, -22.6, -18., -13.7, -10.4, -8.3, -5.9, -2.2,
                2.2, 5.9, 8.3, 10.4, 13.7, 18., 22.6, 26.9],
        13100: [-25.3, -22.4, -18.9, -14.7, -10.4, -7.2, -5.3, -2.2,
                2.2, 5.3, 7.2, 10.4, 14.7, 18.9, 22.4, 25.3],
    }

    tx_pattern_options = {
        "original_16tx": {"acf": original_widebeam_acf_directions, "xcf": original_widebeam_xcf_directions},
        "original_8tx": {"acf": original_8tx_widebeam_acf_directions, "xcf": original_widebeam_xcf_directions},
        "60deg_fov_droopy": {"acf": widebeam_60deg_droop_acf_directions, "xcf": widebeam_60deg_droop_xcf_directions},
        "60deg_fov": {"acf": widebeam_60deg_acf_directions, "xcf": widebeam_60deg_xcf_directions},
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
        tx_pattern = kwargs.get("tx_pattern", None)
        if tx_pattern is None or tx_pattern not in cls.tx_pattern_options.keys():
            raise RuntimeError(f"Require tx_pattern kwarg to use the right beam corrections. "
                               f"Options are {list(cls.tx_pattern_options.keys())}")

        acf_directions = cls.tx_pattern_options[tx_pattern]["acf"]
        xcf_directions = cls.tx_pattern_options[tx_pattern]["xcf"]

        acf_record = copy.deepcopy(record)
        beam_nums = record['beam_nums']
        freq_khz = record['freq']
        old_beam_azms = record['beam_azms']
        acf_beam_azms = np.array([acf_directions[freq_khz][i] for i in beam_nums])
        xcf_beam_azms = np.array([xcf_directions[freq_khz][i] for i in beam_nums])
        acf_record['beam_azms'] = acf_beam_azms
        record['beam_azms'] = xcf_beam_azms

        # Now do the processing with the new beam directions
        acf_record = super().process_record(acf_record)
        record = super().process_record(record)

        record['main_acfs'] = acf_record['main_acfs']

        # The direction of sensitivity is still the same, so revert back to original beam directions.
        record['beam_azms'] = old_beam_azms

        return record