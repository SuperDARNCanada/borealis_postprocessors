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


def beamform_samples(self, filtered_samples, beam_phases):
    """
    Beamform the filtered samples for multiple beams simultaneously.
    :param      filtered_samples:  The filtered input samples.
    :type       filtered_samples:  ndarray [num_slices, num_antennas, num_samples]
    :param      beam_phases:       The beam phases used to phase each antenna's samples before
                                   combining.
    :type       beam_phases:       list
    """

    beam_phases = xp.array(beam_phases)

    # [num_slices, num_antennas, num_samples]
    # [num_slices, num_beams, num_antennas]
    beamformed_samples = xp.einsum('ijk,ilj->ilk', filtered_samples, beam_phases)

    self.beamformed_samples = beamformed_samples


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
    phshift = 2 * math.pi * freq * (((num_antennas-1)/2.0 - antenna) * \
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

    #samples = [sample * amplitude * np.exp(1j * phshift) for sample in basic_samples]
    samples = amplitude * np.exp(1j * phshift) * basic_samples
    return samples


def antiq2bfiq(filename):
    # This is initially designed to only work for the SAS 2019 data fix

    recs = dd.io.load(filename)
    data_directory = os.path.dirname(data_file_path)
    data_file_metadata = filename.split('.')
    date_of_file = data_file_metadata[0]
    timestamp_of_file = '.'.join(data_file_metadata[0:3])
    station_name = data_file_metadata[3]
    slice_id_number = data_file_metadata[4]
    file_suffix = data_file_metadata[-1]

    sorted_keys = sorted(list(recs.keys()))
    tmp_file = filename + ".tmp"

    # Update the file
    print(f'file: {filename}')

    for key_num, group_name in enumerate(sorted_keys):
        cpid = recs[group_name]['experiment_id']
        first_range = 180  #scf.FIRST_RANGE
        first_range_rtt = first_range * 2.0 * 1.0e3 * 1e6 / speed_of_light
        lag_table = list(itertools.combinations(recs[group_name]['pulses'], 2))
        lag_table.append([recs[group_name]['pulses'][0], recs[group_name]['pulses'][0]])  # lag 0
        lag_table = sorted(lag_table, key=lambda x: x[1] - x[0])  # sort by lag number
        lag_table.append([recs[group_name]['pulses'][-1], recs[group_name]['pulses'][-1]])  # alternate lag 0
        recs[group_name]['lags'] = np.array(lag_table, dtype=np.uint32)
        if station_name in ["cly", "rkn", "inv"]:
            num_ranges = 100  # scf.POLARDARN_NUM_RANGES
        elif station_name in ["sas", "pgr"]:
            num_ranges = 75  # scf.STD_NUM_RANGES
        range_sep = 1 / recs[group_name]['rx_sample_rate'] * speed_of_light / 1.0e3 / 2.0

        # Things I think I need
        antenna_spacing = float(config['main_antenna_spacing'])
        intf_antenna_spacing = float(config['interferometer_antenna_spacing'])

        main_beamformed_data = beamform(antennas_data,
                                   recs[group_name]['beam_azms'][()],
                                   recs[group_name]['freq'],
                                   antenna_spacing)
        intf_beamformed_data = beamform(antennas_data,
                                   recs[group_name]['beam_azms'][()],
                                   recs[group_name]['freq'],
                                   antenna_spacing)

    write_dict = {}
    write_dict[group_name] = convert_to_numpy(recs[group_name])
    dd.io.save(tmp_file, write_dict, compression=None)

    # use external h5copy utility to move new record into 2hr file.
    cmd = 'h5copy -i {newfile} -o {twohr} -s {dtstr} -d {dtstr}'
    cmd = cmd.format(newfile=tmp_file, twohr=out_file, dtstr=group_name)

    # Todo: improve call to subprocess.
    sp.call(cmd.split())
    os.remove(tmp_file)


    output_samples_filetype = slice_id_number + ".antennas_iq"
    bfiq_filetype = slice_id_number + ".bfiq"
    rawrf_filetype = "rawrf"
    tx_filetype = "txdata"

    # Choose a record from the provided file, and get that record for each filetype to analyze side by side.
    # Also reshaping data to correct dimensions - if there is a problem with reshaping, we will also not use that record.
    record_data = {}
    try:
        record_data[file_type] = data[file_type][record_name]
        output_samples_iq = record_data[output_samples_filetype]
        number_of_antennas = len(output_samples_iq['antenna_arrays_order'])
        flat_data = np.array(output_samples_iq['data'])
        # reshape to number of antennas (M0..... I3) x nave x number_of_samples
        output_samples_iq_data = np.reshape(flat_data, (number_of_antennas, output_samples_iq['num_sequences'], output_samples_iq['num_samps']))
        output_samples_iq['data'] = output_samples_iq_data
        antennas_present = [int(i.split('_')[-1]) for i in output_samples_iq['antenna_arrays_order']]
        output_samples_iq['antennas_present'] = antennas_present

    except ValueError as e:
        print('Record {} raised an exception in filetype {}:\n'.format(record_name, file_type))

    beamforming_dict = {}
    # print('BEAM AZIMUTHS: {}'.format(beam_azms))
    for sequence_num in range(0, nave):
        # print('SEQUENCE NUMBER {}'.format(sequence_num))
        sequence_dict = beamforming_dict[sequence_num] = {}
        for filetype, record_dict in record_data.items():
            print(filetype)
            sequence_filetype_dict = sequence_dict[filetype] = {}
            data_description_list = list(record_dict['data_descriptors'])

            sequence_filetype_dict['decimated_data'] = decimated_data

            # STEP 2: BEAMFORM ANY UNBEAMFORMED DATA
            if filetype != bfiq_filetype:
                # need to beamform the data.
                antenna_list = []
                # print(decimated_data.shape)
                if data_description_list == ['num_antennas', 'num_sequences', 'num_samps']:
                    for antenna in range(0, record_dict['data'].shape[0]):
                        antenna_list.append(decimated_data[antenna, :])
                    antenna_list = np.array(antenna_list)
                elif data_description_list == ['num_sequences', 'num_antennas', 'num_samps']:
                    for antenna in range(0, record_dict['data'].shape[1]):
                        antenna_list.append(decimated_data[antenna, :])
                    antenna_list = np.array(antenna_list)
                else:
                    raise Exception('Not sure how to beamform with the dimensions of this data: {}'.format(
                        record_dict['data_descriptors']))

                # beamform main array antennas only.
                main_antennas_mask = (record_dict['antennas_present'] < main_antenna_count)
                intf_antennas_mask = (record_dict['antennas_present'] >= main_antenna_count)
                decimated_beamformed_data = beamform(antenna_list[main_antennas_mask][:].copy(),
                                                     beam_azms, freq, antenna_spacing)  # TODO test
                # without
                # .copy()
                intf_decimated_beamformed_data = beamform(antenna_list[intf_antennas_mask][:].copy(),
                                                          beam_azms, freq, intf_antenna_spacing)
            # else:
            #     decimated_beamformed_data = decimated_data
            #     intf_decimated_beamformed_data = intf_decimated_data

            sequence_filetype_dict['main_bf_data'] = decimated_beamformed_data  # this has 2 dimensions: num_beams x num_samps for this sequence.
            sequence_filetype_dict['intf_bf_data'] = intf_decimated_beamformed_data


if __name__ == '__main__':
    parser = testing_parser()
    args = parser.parse_args()
    data_file_path = args.filename
    data_file = os.path.basename(data_file_path)
    antiq2bfiq(filename)
