# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains functions to image antennas_iq data into
rawacf-style data with higher azimuthal resolution than
standard SuperDARN rawacf files.
"""


def image_data(infile, outfile, num_bins, min_angle, max_angle):
    """
    Performs imaging on data from infile, for num_bins azimuthal angles
    between min_angle and max_angle.

    :param infile:      Name of antennas_iq file to image. String
    :param outfile:     Name of file where imaged data will be stored. String
    :param num_bins:    Number of azimuthal bins. Int
    :param min_angle:   Minimum angle to right of boresight. Degrees
    :param max_angle:   Maximum angle to right of boresight. Degrees
    :return:
    """
