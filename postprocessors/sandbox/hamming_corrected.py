# Copyright 2021 SuperDARN Canada, University of Saskatchewan

"""
This file contains functions for converting antennas_iq files from widebeam experiments
to rawacf files, using a Hamming window in amplitude for beamforming to reduce receiver sidelobes.
Receiver beam directions are adjusted to correct for azimuthal sensitivity variations.
"""
import copy
from collections import OrderedDict
import numpy as np

from postprocessors.sandbox.hamming_beams import HammingWindowBeamforming


class HammingBeamformingCorrected(HammingWindowBeamforming):
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

    acf_directions = {
        10400: [-26.9, -22.66, -19.22, -14.38, -11.34, -7.7, -3.76, -1.52,
                1.52, 3.76, 7.8, 11.34, 14.38, 19.22, 22.66, 26.9],
        10500: [-27., -22.56, -19.22, -14.48, -11.34, -7.8, -3.76, -1.52,
                1.52, 3.76, 7.8, 11.34, 14.48, 19.22, 22.56, 27.],
        10600: [-27.1, -22.46, -19.22, -14.48, -11.32, -7.8, -3.76, -1.46,
                1.46, 3.76, 7.8, 11.32, 14.48, 19.22, 22.46, 27.1],
        10700: [-27.2, -22.46, -19.32, -14.58, -11.27, -7.9, -3.76, -1.46,
                1.46, 3.76, 7.9, 11.27, 14.58, 19.32, 22.46, 27.3],
        10800: [-27.3, -22.46, -19.32, -14.68, -11.24, -7.9, -3.76, -1.46,
                1.46, 3.76, 7.9, 11.24, 14.68, 19.32, 22.46, 27.4],
        10900: [-27.4, -22.36, -19.32, -14.78, -11.24, -7.9, -3.76, -1.46,
                1.46, 3.76, 8., 11.24, 14.78, 19.32, 22.36, 27.5],
        12200: [-26.6, -21.66, -17.42, -14.84, -11.24, -8.1, -4.86, -1.82,
                1.82, 4.86, 8.1, 11.24, 14.84, 17.42, 21.66, 26.7],
        12300: [-26.7, -21.76, -17.52, -14.78, -11.24, -8.1, -4.86, -1.82,
                1.82, 4.86, 8.1, 11.24, 14.78, 17.52, 21.76, 26.8],
        12500: [-26.9, -21.86, -17.52, -14.74, -11.34, -7.9, -4.86, -1.92,
                1.92, 4.86, 7.9, 11.34, 14.74, 17.52, 21.86, 27.],
        13000: [-27.3, -22.06, -17.52, -14.64, -11.24, -7.7, -4.96, -2.12,
                2.12, 4.96, 7.7, 11.24, 14.64, 17.52, 22.06, 27.5],
        13100: [-27.7, -22.06, -17.62, -14.54, -11.44, -7.6, -4.93, -2.12,
                2.12, 4.93, 7.6, 11.44, 14.54, 17.62, 22.06, 27.8],
        13200: [-27.7, -22.16, -17.72, -14.54, -11.34, -7.6, -4.96, -2.12,
                2.12, 4.96, 7.6, 11.34, 14.54, 17.72, 22.16, 27.8],
    }
    xcf_directions = {
        10400: [-28.8, -23.96, -19.92, -14.68, -11.24, -7.4, -3.48, -1.36,
                1.36, 3.48, 7.5, 11.24, 14.68, 19.92, 23.96, 28.8],
        10500: [-28.8, -23.86, -20.02, -14.78, -11.24, -7.5, -3.53, -1.41,
                1.41, 3.53, 7.5, 11.24, 14.78, 20.02, 23.86, 29.],
        10600: [-29., -23.76, -20.02, -14.78, -11.24, -7.5, -3.48, -1.32,
                1.32, 3.48, 7.6, 11.24, 14.78, 20.02, 23.76, 29.],
        10700: [-29.1, -23.76, -20.12, -14.88, -11.24, -7.6, -3.53, -1.32,
                1.32, 3.53, 7.6, 11.24, 14.88, 20.12, 23.76, 29.2],
        10800: [-29.3, -23.76, -20.12, -14.98, -11.24, -7.6, -3.53, -1.32,
                1.32, 3.53, 7.7, 11.24, 14.98, 20.12, 23.76, 29.4],
        10900: [-29.3, -23.66, -20.12, -15.08, -11.24, -7.7, -3.56, -1.32,
                1.32, 3.56, 7.7, 11.24, 15.08, 20.12, 23.66, 29.4],
        12200: [-28.4, -22.26, -17.62, -14.78, -11.24, -8.15, -4.96, -1.82,
                1.82, 4.96, 8.15, 11.24, 14.78, 17.62, 22.26, 28.5],
        12300: [-28.5, -22.46, -17.62, -14.78, -11.24, -8.1, -4.96, -1.82,
                1.82, 4.96, 8.1, 11.24, 14.78, 17.62, 22.46, 28.6],
        12500: [-28.7, -22.56, -17.62, -14.69, -11.24, -7.9, -4.88, -1.92,
                1.92, 4.88, 7.9, 11.24, 14.69, 17.62, 22.56, 28.8],
        13000: [-29.3, -22.86, -17.72, -14.58, -11.14, -7.6, -4.96, -2.12,
                2.12, 4.96, 7.6, 11.14, 14.58, 17.72, 22.86, 29.4],
        13100: [-29.6, -22.96, -17.82, -14.48, -11.34, -7.55, -4.93, -2.12,
                2.12, 4.93, 7.55, 11.34, 14.48, 17.82, 22.96, 29.7],
        13200: [-29.6, -23.06, -17.92, -14.48, -11.24, -7.6, -4.96, -2.12,
                2.12, 4.96, 7.6, 11.24, 14.48, 17.92, 23.06, 29.7],
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

