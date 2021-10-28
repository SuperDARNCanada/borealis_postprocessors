# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains functions for converting antennas_iq files
to bfiq files.
"""

try:
    import cupy as xp
except ImportError:
    import numpy as xp
    cupy_available = False
else:
    cupy_available = True


def beamform_samples(self, samples, beam_phases):
    """
    Beamform the samples for multiple beams simultaneously.

    :param      samples:           The filtered input samples.
    :type       samples:           ndarray [num_slices, num_antennas, num_samples]
    :param      beam_phases:       The beam phases used to phase each antenna's samples before
                                   combining.
    :type       beam_phases:       list

    """

    beam_phases = xp.array(beam_phases)

    # [num_slices, num_antennas, num_samples]
    # [num_slices, num_beams, num_antennas]
    beamformed_samples = xp.einsum('ijk,ilj->ilk', samples, beam_phases)

    return beamformed_samples

