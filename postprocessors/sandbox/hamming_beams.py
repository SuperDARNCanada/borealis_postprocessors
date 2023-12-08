# Copyright 2021 SuperDARN Canada, University of Saskatchewan

"""
This file contains functions for converting antennas_iq files from widebeam experiments
to rawacf files, using a Hamming window in amplitude for beamforming to reduce receiver sidelobes.
"""
from typing import Union
import numpy as np

try:
    import cupy as xp
except ImportError:
    import numpy as xp

    cupy_available = False
else:
    cupy_available = True

from postprocessors import AntennasIQ2Rawacf, AntennasIQ2Bfiq


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
        The write structure of the file. Structures include:
        'array'
        'site'
    outfile_structure: str
        The desired structure of the output file. Same structures as above, plus 'dmap'.
    beam_azms: list[float]
        List of all beam directions (in degrees) to reprocess into
    beam_nums: list[uint]
        List describing beam order. Numbers in this list correspond to indices of beam_azms
    """
    hamming_window = [0.08081232549588463, 0.12098514265395757, 0.23455777475180511, 0.4018918165398586,
                      0.594054435182454, 0.7778186328978896, 0.9214100134552521, 1.0,
                      1.0, 0.9214100134552521, 0.7778186328978896, 0.594054435182454,
                      0.4018918165398586, 0.23455777475180511, 0.12098514265395757, 0.08081232549588463]

    def __init__(self, infile: str, outfile: str, infile_structure: str, outfile_structure: str,
                 beam_azms: Union[list, None] = None, beam_nums: Union[list, None] = None):
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

        # Use default 16-beam arrangement if no beams are specified
        if beam_azms is None:
            # Add extra phases here
            # STD_16_BEAM_ANGLE from superdarn_common_fields
            self.beam_azms = [-24.3, -21.06, -17.82, -14.58, -11.34, -8.1, -4.86, -1.62, 1.62, 4.86, 8.1, 11.34, 14.58,
                              17.82, 21.06, 24.3]

            # STD_16_FORWARD_BEAM_ORDER
            self.beam_nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        else:
            self.beam_azms = beam_azms
            # Step through beam_azms sequentially if not specified
            if beam_nums is None:
                self.beam_nums = range(len(beam_azms))
            elif len(beam_nums) != len(beam_azms):
                raise ValueError('beam_nums should be same length as beam_azms.')
            else:
                self.beam_nums = beam_nums

        self.process_file(beam_azms=self.beam_azms, beam_nums=self.beam_nums)

    @classmethod
    def beamform(cls, antennas_data: np.array, beamdirs: np.array, rxfreq: float, ants_in_array: int,
                 antenna_spacing: float,
                 antenna_indices: np.array) -> np.array:
        """
        Beamforms the data from each antenna and sums to create one dataset for each beam direction.
        Overwrites the implementation in AntennasIQ2Bfiq to use an amplitude taper

        Parameters
        ----------
        antennas_data: np.array
            Numpy array of dimensions num_antennas x num_samps. All antennas are assumed to be from the same array
            and are assumed to be side by side with uniform antenna spacing
        beamdirs: np.array
            Azimuthal beam directions in degrees off boresight
        rxfreq: float
            Frequency of the received beam
        ants_in_array: int
            Number of physical antennas in the array
        antenna_spacing: float
            Spacing in metres between antennas (assumed uniform)
        antenna_indices: np.array
            Mapping of antenna channels to physical antennas in the uniformly spaced array.

        Returns
        -------
        beamformed_data: np.array
            Array of shape [num_beams, num_samps]
        """
        beamformed_data = []

        # [num_antennas, num_samps]
        num_antennas, num_samps = antennas_data.shape

        # Loop through all beam directions
        for beam_direction in beamdirs:
            antenna_phase_shifts = []

            # Get phase shift for each antenna
            for antenna in antenna_indices:
                phase_shift = AntennasIQ2Bfiq.get_phshift(beam_direction, rxfreq, antenna, ants_in_array,
                                                          antenna_spacing)
                # Bring into range (-2*pi, 2*pi)
                antenna_phase_shifts.append(phase_shift)

            # Apply phase shift to data from respective antenna
            if num_antennas == 16:
                phased_antenna_data = [AntennasIQ2Bfiq.shift_samples(antennas_data[i], antenna_phase_shifts[i],
                                                                     cls.hamming_window[i])
                                       for i in range(num_antennas)]
            else:
                phased_antenna_data = [AntennasIQ2Bfiq.shift_samples(antennas_data[i], antenna_phase_shifts[i], 1.0)
                                       for i in range(num_antennas)]
            phased_antenna_data = xp.array(phased_antenna_data)

            # Sum across antennas to get beamformed data
            one_beam_data = xp.sum(phased_antenna_data, axis=0)
            beamformed_data.append(one_beam_data)
        beamformed_data = xp.array(beamformed_data)

        return beamformed_data
