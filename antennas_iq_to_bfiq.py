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
import borealis
import deepdish as dd
import batch_log

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


def antiq2bfiq(filename):
    data_directory = os.path.dirname(data_file_path)
    antenna_spacing = float(config['main_antenna_spacing'])
    intf_antenna_spacing = float(config['interferometer_antenna_spacing'])

    data_file_metadata = data_file.split('.')

    date_of_file = data_file_metadata[0]
    timestamp_of_file = '.'.join(data_file_metadata[0:3])
    station_name = data_file_metadata[3]
    slice_id_number = data_file_metadata[4]

    file_suffix = data_file_metadata[-1]

    if file_suffix not in ['hdf5', 'site']:
        raise Exception('Incorrect File Suffix: {}'.format(file_suffix))

    if file_suffix == 'hdf5':
        type_of_file = data_file_metadata[-2]  # XX.hdf5
    else:  # site
        type_of_file = data_file_metadata[-3]  # XX.hdf5.site
        file_suffix = data_file_metadata[-2] + '.' + data_file_metadata[-1]

    if type_of_file == slice_id_number:
        slice_id_number = '0'  # choose the first slice to search for other available files.
    else:
        type_of_file = slice_id_number + '.' + type_of_file

    output_samples_filetype = slice_id_number + ".antennas_iq"
    bfiq_filetype = slice_id_number + ".bfiq"
    rawrf_filetype = "rawrf"
    tx_filetype = "txdata"
    file_types_avail = [bfiq_filetype, output_samples_filetype, tx_filetype, rawrf_filetype]

    if type_of_file not in file_types_avail:
        raise Exception(
            'Data type: {} not incorporated in script. Allowed types: {}'.format(type_of_file,
                                                                                 file_types_avail))

    data = {}
    for file_type in list(file_types_avail):  # copy of file_types_avail so we can modify it within.
        try:
            filename = data_directory + '/' + timestamp_of_file + \
                       '.' + station_name + '.' + file_type + '.' + file_suffix
            data[file_type] = dd.io.load(filename)
        except:
            file_types_avail.remove(file_type)
            if file_type == type_of_file:  # if this is the filename you provided.
                raise

    # Choose a record from the provided file, and get that record for each filetype to analyze side by side.
    # Also reshaping data to correct dimensions - if there is a problem with reshaping, we will also not use that record.
    good_record_found = False
    record_attempts = 0
    while not good_record_found:
        if args.record:
            record_name = args.record
        else:
            record_name = random.choice(list(data[type_of_file].keys()))
        print('Record Name: {}'.format(record_name))

        record_data = {}

        try:
            for file_type in file_types_avail:
                record_data[file_type] = data[file_type][record_name]
                # antennas_iq
                if file_type == output_samples_filetype:
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
            traceback.print_exc()
            print('\nA new record will be selected.')
            record_attempts += 1
            if record_attempts == 3:
                print('FILES FAILED WITH 3 FAILED ATTEMPTS TO LOAD RECORDS.')
                raise  # something is wrong with the files
        else:  # no errors
            good_record_found = True

    beamforming_dict = {}
    # print('BEAM AZIMUTHS: {}'.format(beam_azms))
    for sequence_num in range(0, nave):
        # print('SEQUENCE NUMBER {}'.format(sequence_num))
        sequence_dict = beamforming_dict[sequence_num] = {}
        for filetype, record_dict in record_data.items():
            print(filetype)
            sequence_filetype_dict = sequence_dict[filetype] = {}
            data_description_list = list(record_dict['data_descriptors'])
            # STEP 1: DECIMATE IF NECESSARY
            if not math.isclose(record_dict['rx_sample_rate'], decimated_rate, abs_tol=0.001):
                # print(decimated_rate)
                # print(record_dict['rx_sample_rate'])
                # we aren't at 3.3 kHz - need to decimate.
                # print(record_dict['rx_sample_rate'])
                dm_rate = int(record_dict['rx_sample_rate'] / decimated_rate)
                # print(dm_rate)
                dm_start_sample = record_dict['dm_start_sample']
                dm_end_sample = -1 - dm_start_sample  # this is the filter size
                if data_description_list == ['num_antenna_arrays', 'num_sequences', 'num_beams', 'num_samps']:
                    decimated_data = record_dict['data'][0][sequence_num][:][
                                     dm_start_sample:dm_end_sample:dm_rate]  # grab only main array data, first sequence, all beams.
                    intf_decimated_data = record_dict['data'][1][sequence_num][:][dm_start_sample:dm_end_sample:dm_rate]
                elif data_description_list == ['num_antennas', 'num_sequences', 'num_samps']:
                    decimated_data = record_dict['data'][:, sequence_num,
                                     dm_start_sample:dm_end_sample:dm_rate]  # all antennas.
                elif data_description_list == ['num_sequences', 'num_antennas', 'num_samps']:
                    if filetype == tx_filetype:  # tx data has sequence number 0 for all
                        decimated_data = record_dict['data'][0, :, dm_start_sample:dm_end_sample:dm_rate]
                    else:
                        decimated_data = record_dict['data'][sequence_num, :, dm_start_sample:dm_end_sample:dm_rate]
                else:
                    raise Exception('Not sure how to decimate with the dimensions of this data: {}'.format(
                        record_dict['data_descriptors']))

            else:
                if data_description_list == ['num_antenna_arrays', 'num_sequences', 'num_beams', 'num_samps']:
                    decimated_data = record_dict['data'][0, sequence_num, :, :]  # only main array
                    intf_decimated_data = record_dict['data'][1, sequence_num, :, :]
                elif data_description_list == ['num_antennas', 'num_sequences', 'num_samps']:
                    decimated_data = record_dict['data'][:, sequence_num, :]  # first sequence only, all antennas.
                elif data_description_list == ['num_sequences', 'num_antennas', 'num_samps']:
                    if filetype == tx_filetype:
                        decimated_data = record_dict['data'][0, :, :]  # first sequence only, all antennas.
                    else:
                        decimated_data = record_dict['data'][sequence_num, :, :]  # first sequence only, all antennas.
                else:
                    raise Exception('Unexpected data dimensions: {}'.format(record_dict['data_descriptors']))

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
            else:
                decimated_beamformed_data = decimated_data
                intf_decimated_beamformed_data = intf_decimated_data

            sequence_filetype_dict[
                'main_bf_data'] = decimated_beamformed_data  # this has 2 dimensions: num_beams x num_samps for this sequence.
            sequence_filetype_dict['intf_bf_data'] = intf_decimated_beamformed_data

            # STEP 3: FIND THE PULSES IN THE DATA
            for beamnum in range(0, sequence_filetype_dict['main_bf_data'].shape[0]):

                len_of_data = sequence_filetype_dict['main_bf_data'].shape[1]
                pulse_indices = find_pulse_indices(sequence_filetype_dict['main_bf_data'][beamnum], 0.3)
                if len(pulse_indices) > len(pulses):  # sometimes we get two samples from the same pulse.
                    if math.fmod(len(pulse_indices), len(pulses)) == 0.0:
                        step_size = int(len(pulse_indices) / len(pulses))
                        pulse_indices = pulse_indices[step_size - 1::step_size]

                pulse_points = [False if i not in pulse_indices else True for i in range(0, len_of_data)]
                sequence_filetype_dict['pulse_indices'] = pulse_indices

                # verify pulse indices make sense.
                # tau_spacing is in microseconds
                num_samples_in_tau_spacing = int(round(tau_spacing * 1.0e-6 * decimated_rate))
                pulse_spacing = pulses * num_samples_in_tau_spacing
                expected_pulse_indices = list(pulse_spacing + pulse_indices[0])
                if expected_pulse_indices != pulse_indices:
                    sequence_filetype_dict['calculate_offsets'] = False
                    print(expected_pulse_indices)
                    print(pulse_indices)
                    print('Pulse Indices are Not Equal to Expected for filetype {} sequence {}'.format(filetype,
                                                                                                       sequence_num))
                    print('Phase Offsets Cannot be Calculated for this filetype {} sequence {}'.format(filetype,
                                                                                                       sequence_num))
                else:
                    sequence_filetype_dict['calculate_offsets'] = True

                # get the phases of the pulses for this data.
                pulse_data = sequence_filetype_dict['main_bf_data'][beamnum][pulse_points]
                sequence_filetype_dict['pulse_samples'] = pulse_data
                pulse_phases = np.angle(pulse_data) * 180.0 / math.pi
                sequence_filetype_dict['pulse_phases'] = pulse_phases



if __name__ == '__main__':
    parser = testing_parser()
    args = parser.parse_args()
    data_file_path = args.filename
    data_file = os.path.basename(data_file_path)
    antiq2bfiq(filename)
