# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Authors: Adam Lozinsky
"""
This file contains a function for processing antennas IQ data into beam-formed IQ
data product matching the real-time product produced by datawrite module in Borealis.
"""

import argparse
import bz2
import os
import itertools
import subprocess as sp
import numpy as np
import warnings
import tables
import pydarnio
import batch_log

import deepdish as dd
from scipy.constants import speed_of_light
import math

warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)


def usage_msg():
    """
    Return the usage message for this process.

    This is used if a -h flag or invalid arguments are provided.
    """

    usage_message = """ borealis_fixer.py [-h] filename fixed_dat_dir

    **** NOT TO BE USED IN PRODUCTION ****
    **** USE WITH CAUTION ****
    Modify a borealis file with updated data fields. Modify the script where
    indicated to update the file. Used in commissioning phase of Borealis when 
    data fields were not finalized."""

    return usage_message


def script_parser():
    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("filename", help="Path to the file that you wish to modify")
    parser.add_argument("fixed_data_dir", help="Path to place the updated file in.")

    return parser


def beamform_samples(filtered_samples, beam_phases):
    """
    Beamform the filtered samples for multiple beams simultaneously.
    :param      filtered_samples:  The filtered input samples.
    :type       filtered_samples:  ndarray [num_slices, num_antennas, num_samples]
    :param      beam_phases:       The beam phases used to phase each antenna's samples before
                                   combining.
    :type       beam_phases:       list
    """

    beam_phases = np.array(beam_phases)
    beamformed_samples = np.einsum('ijk,ilj->ilk', filtered_samples, beam_phases)

    return beamformed_samples


def beamform(antennas_data, beamdirs, rxfreq, antenna_spacing):
    """
    :param antennas_data: numpy array of dimensions num_antennas x num_samps. All antennas are assumed to be
    from the same array and are assumed to be side by side with antenna spacing 15.24 m, pulse_shift = 0.0
    :param beamdirs: list of azimuthal beam directions in degrees off boresite
    :param rxfreq: frequency to beamform at.
    :param antenna_spacing: spacing in metres between antennas, used to get the phase shift that
    corresponds to an azimuthal direction.
    """

    beamformed_data = []
    for beam_direction in beamdirs:
        antenna_phase_shifts = []
        for antenna in range(0, antennas_data.shape[0]):
            phase_shift = math.fmod((-1 * get_phshift(beam_direction, rxfreq, antenna, 0.0,
                                     antennas_data.shape[0], antenna_spacing)), 2*math.pi)
            antenna_phase_shifts.append(phase_shift)
        phased_antenna_data = [shift_samples(antennas_data[i], antenna_phase_shifts[i], 1.0) for i in range(0, antennas_data.shape[0])]
        phased_antenna_data = np.array(phased_antenna_data)
        one_beam_data = np.sum(phased_antenna_data, axis=0)
        beamformed_data.append(one_beam_data)
    beamformed_data = np.array(beamformed_data)

    return beamformed_data


def get_phshift(beamdir, freq, antenna, pulse_shift, num_antennas, antenna_spacing,
        centre_offset=0.0):
    """
    Find the phase shift for a given antenna and beam direction.
    Form the beam given the beam direction (degrees off boresite), the tx frequency, the antenna number,
    a specified extra phase shift if there is any, the number of antennas in the array, and the spacing
    between antennas.
    :param beamdir: the azimuthal direction of the beam off boresight, in degrees, positive beamdir being to
        the right of the boresight (looking along boresight from ground). This is for this antenna.
    :param freq: transmit frequency in kHz
    :param antenna: antenna number, INDEXED FROM ZERO, zero being the leftmost antenna if looking down the boresight
        and positive beamdir right of boresight
    :param pulse_shift: in degrees, for phase encoding
    :param num_antennas: number of antennas in this array
    :param antenna_spacing: distance between antennas in this array, in meters
    :param centre_offset: the phase reference for the midpoint of the array. Default = 0.0, in metres.
     Important if there is a shift in centre point between arrays in the direction along the array.
     Positive is shifted to the right when looking along boresight (from the ground).
    :returns phshift: a phase shift for the samples for this antenna number, in radians.
    """
    freq = freq * 1000.0  # convert to Hz.

    beamdir = float(beamdir)

    beamrad = math.pi * float(beamdir) / 180.0

    # Pointing to right of boresight, use point in middle (hypothetically antenna 7.5) as phshift=0
    phshift = 2 * math.pi * freq * (((num_antennas-1)/2.0 - antenna) *
        antenna_spacing + centre_offset) * math.cos(math.pi / 2.0 - beamrad) \
        / speed_of_light

    # Add an extra phase shift if there is any specified
    phshift = phshift + math.radians(pulse_shift)

    phshift = math.fmod(phshift, 2 * math.pi)

    return phshift


def shift_samples(basic_samples, phshift, amplitude):
    """
    Shift samples for a pulse by a given phase shift.
    Take the samples and shift by given phase shift in rads and adjust amplitude as
    required for imaging.
    :param basic_samples: samples for this pulse, numpy array
    :param phshift: phase for this antenna to offset by in rads, float
    :param amplitude: amplitude for this antenna (= 1 if not imaging), float
    :returns samples: basic_samples that have been shaped for the antenna for the
     desired beam.
    """
    samples = amplitude * np.exp(1j * phshift) * basic_samples

    return samples


def beamform_file(filename, out_file):
    # This is initially designed to only work for the SAS 2019 data fix
    # Input a antennas_iq.site file, transform to a bfiq.site file, then save as a bfiq.array file.
    # The last step simply need to convert a temp file to array format using pydarnio then delete the temp.

    recs = dd.io.load(filename)
    data_file_metadata = filename.split('.')
    station_name = data_file_metadata[3]
    sorted_keys = sorted(list(recs.keys()))
    tmp_file = filename + ".tmp"

    def convert_to_numpy(data):
        """Converts lists stored in dict into numpy array. Recursive.
        Args:
            data (Python dictionary): Dictionary with lists to convert to numpy arrays.
        """
        for k, v in data.items():
            if isinstance(v, dict):
                convert_to_numpy(v)
            elif isinstance(v, list):
                data[k] = np.array(v)
            else:
                continue
        return data

    def check_dataset_add(k, v):
        if k not in recs[group_name].keys():
            recs[group_name][k] = v
            if key_num == 0:
                print(f'\t- added: {k}')

    def check_dataset_rename(k, v):
        if k in recs[group_name].keys():
            recs[group_name][v] = recs[group_name][k]
            del recs[group_name][k]
            if key_num == 0:
                print(f'\t- updated: {k}')

    def check_dataset_del(k):
        if k in recs[group_name].keys():
            del recs[group_name][k]
            if key_num == 0:
                print(f'\t- removed: {k}')

                if 'timestamp_of_write' in recs[group_name].keys():
                    del recs[group_name]['timestamp_of_write']
                    if key_num == 0:
                        print('timestamp_of_write removed')

    def check_dataset_revalue(k, v):
        if k in recs[group_name].keys():
            recs[group_name][k] = v
            if key_num == 0:
                print(f'\t- updated: {k}')

    # Update the file
    print(f'file: {filename}')

    for key_num, group_name in enumerate(sorted_keys):
        # Find all the bfiq required missing datasets or create them

        # first_range
        first_range = 180.0  #scf.FIRST_RANGE
        check_dataset_add('first_range', np.float32(first_range))

        # first_range_rtt
        first_range_rtt = first_range * 2.0 * 1.0e3 * 1e6 / speed_of_light
        check_dataset_add('first_range_rtt', np.float32(first_range_rtt))

        # lags
        lag_table = list(itertools.combinations(recs[group_name]['pulses'], 2))
        lag_table.append([recs[group_name]['pulses'][0], recs[group_name]['pulses'][0]])  # lag 0
        lag_table = sorted(lag_table, key=lambda x: x[1] - x[0])  # sort by lag number
        lag_table.append([recs[group_name]['pulses'][-1], recs[group_name]['pulses'][-1]])  # alternate lag 0
        lags = np.array(lag_table, dtype=np.uint32)
        check_dataset_add('lags', lags)

        # num_ranges
        if station_name in ["cly", "rkn", "inv"]:
            num_ranges = 100  # scf.POLARDARN_NUM_RANGES
            check_dataset_add('num_ranges', np.uint32(num_ranges))
        elif station_name in ["sas", "pgr"]:
            num_ranges = 75  # scf.STD_NUM_RANGES
            check_dataset_add('num_ranges', np.uint32(num_ranges))

        # range_sep
        range_sep = 1 / recs[group_name]['rx_sample_rate'] * speed_of_light / 1.0e3 / 2.0
        check_dataset_add('range_sep', np.float32(range_sep))

        # Check pulse_phase_offset type
        recs[group_name]['pulse_phase_offset'] = np.float32(recs[group_name]['pulse_phase_offset'][()])

        # Beamform the data
        main_antenna_spacing = 15.24  # For SAS from config file
        intf_antenna_spacing = 15.24  # For SAS from config file
        beam_azms = recs[group_name]['beam_azms'][()]
        freq = recs[group_name]['freq']

        # antennas data shape  = [num_antennas, num_sequences, num_samps]
        antennas_data = recs[group_name]['data']
        antennas_data = antennas_data.reshape(recs[group_name]['data_dimensions'])
        main_beamformed_data = np.array([], dtype=np.complex64)
        intf_beamformed_data = np.array([], dtype=np.complex64)

        # Loop through every sequence and beamform the data
        # output shape after loop is [num_sequences, num_beams, num_samps]
        for j in range(antennas_data.shape[1]):
            # data input shape = [num_antennas, num_samps]
            # data return shape = [num_beams, num_samps]
            main_beamformed_data = np.append(main_beamformed_data,
                                             beamform(antennas_data[0:16, j, :], beam_azms, freq, main_antenna_spacing))
            intf_beamformed_data = np.append(intf_beamformed_data,
                                             beamform(antennas_data[16:20, j, :], beam_azms, freq, intf_antenna_spacing))

        # Remove iq data for bfiq data.
        # Data shape after append is [num_antenna_arrays, num_sequences, num_beams, num_samps]
        # Then flatten the array for final .site format
        del recs[group_name]['data']
        recs[group_name]['data'] = np.append(main_beamformed_data, intf_beamformed_data).flatten()

        # data_dimensions
        # We need set num_antennas_arrays=2 for two arrays and num_beams=length of beam_azms
        data_dimensions = recs[group_name]['data_dimensions']
        recs[group_name]['data_dimensions'] = np.array([2, data_dimensions[1], len(beam_azms), data_dimensions[2]], dtype=np.uint32)

        # NOTE (Adam): After all this we essentially could loop through all records and build the array file live but,
        # it is just as easy to save the .site format and use pydarnio to reload the data, verify its contents
        # automatically and then reshape it into .array format (which automatically handles all the zero padding).

        write_dict = {}
        write_dict[group_name] = convert_to_numpy(recs[group_name])
        dd.io.save(tmp_file, write_dict, compression=None)

        # use external h5copy utility to move new record into 2hr file.
        cmd = 'h5copy -i {newfile} -o {twohr} -s {dtstr} -d {dtstr}'
        cmd = cmd.format(newfile=tmp_file, twohr=out_file + '.tmp', dtstr=group_name)

    bfiq_reader = pydarnio.BorealisRead(tmp_file, 'bfiq', 'site')
    array_data = bfiq_reader.arrays
    bfiq_writer = pydarnio.BorealisWrite(out_file, array_data, 'bfiq', 'array')

    os.remove(tmp_file)
    os.remove(out_file + '.tmp')


def antiq2bfiq(filename, fixed_data_dir=''):
    """
    Checks if the file is bz2, decompresses if necessary, and
    writes to a fixed data directory. If the file was bz2, then the resulting
    file will also be compressed to bz2.
    Parameters
    ----------
    filename
        filename to update, can be bz2 compressed
    fixed_data_dir
        pathname to put the new file into
    """

    out_file = os.path.basename(filename).split('.')
    out_file = '.'.join(out_file[0:5]) + '.bfiq.hdf5.array'

    if fixed_data_dir == '/':
        out_file = fixed_data_dir + out_file
    elif fixed_data_dir == '':
        out_file = fixed_data_dir + out_file
    else:
        out_file = fixed_data_dir + "/" + out_file

    beamform_file(filename, out_file)

    return out_file


if __name__ == '__main__':

    # Todo (Adam): Need to make this tool properly callable from cli and not require a list.txt of
    #              files made from batch_log.py. Although, this may be a one-off tool.

    log_file = 'processed_antiq_files.txt'
    files = batch_log.read_file(log_file)
    for file in files:
        path = os.path.dirname(file).split('/')
        path = '/'.join(path[0:-2]) + '/sas_2019_processed/' + path[-1] + '/'
        antiq2bfiq(file, path)
