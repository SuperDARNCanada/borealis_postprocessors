# This is intended for v0.4 output_ptrs specifically the sas 2019 data.
# This is just a simple script for single time use.
import bz2
import os
import itertools
import subprocess as sp
import numpy as np
import warnings
import tables
import deepdish as dd
import batch_log

warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)


def update_file(filename, out_file):
    recs = dd.io.load(filename)
    sorted_keys = sorted(list(recs.keys()))
    tmp_file = out_file + ".tmp"

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
        # v0.4- datasets;
        # antenna_arrays_order
        # beam_azms
        # beam_nums
        # blanked_samples
        sample_spacing = int(recs[group_name]['tau_spacing'] / recs[group_name]['tx_pulse_len'])
        blanked = recs[group_name]['pulses'] * sample_spacing
        blanked = np.sort(np.concatenate((blanked, blanked + 1)))
        check_dataset_add('blanked_samples', blanked)
        # borealis_git_hash
        # data
        # data_descriptors
        # data_dimensions
        # data_normalization_factor
        check_dataset_add('data_normalization_factor', np.float64(9999999.999999996))
        # experiment_comment
        check_dataset_rename('comment', 'experiment_comment')
        # experiment_id
        check_dataset_revalue('experiment_id', np.int64(recs[group_name]['experiment_id']))
        # experiment_name
        check_dataset_rename('experiment_string', 'experiment_name')
        # freq
        # int_time
        # intf_antenna_count
        # main_antenna_count
        # noise_at_freq
        check_dataset_add('noise_at_freq', np.array([0.0] * int(recs[group_name]['num_sequences']), dtype=np.float64))
        # num_samps
        # num_sequences
        # num_slices
        check_dataset_add('num_slices', np.int64(1))
        # pulse_phase_offset
        check_dataset_add('pulse_phase_offset', np.zeros(recs[group_name]['pulses'].shape[0], dtype=np.int))
        # pulses
        # rx_sample_rate
        # sample_data_type
        # scar_start_marker
        # slice_comment
        check_dataset_add('slice_comment', np.unicode_(''))
        # sqn_timestamps
        # station
        # tau_spacing
        # tx_pulse_len

        # v0.5+ datasets;
        # slice_interfacing
        # scheduling_mode
        # slice_id

        # Must be removed datasets;
        # timestamp_of_write
        check_dataset_del('timestamp_of_write')
        # intf_acfs
        check_dataset_del('intf_acfs')
        # main_acfs
        check_dataset_del('main_acfs')
        # xcfs
        check_dataset_del('xcfs')
        # num_pulses
        check_dataset_del('num_pulses')

        # bfiq datasets; These are generated at the beamform() step and should not be in antennas_iq files.
        # first_range
        check_dataset_del('first_range')
        # first_range_rtt
        check_dataset_del('first_range_rtt')
        # lags
        check_dataset_del('lags')
        # num_ranges
        # range_sep
        check_dataset_del('range_sep')

        write_dict = {}
        write_dict[group_name] = convert_to_numpy(recs[group_name])
        dd.io.save(tmp_file, write_dict, compression=None)

        # use external h5copy utility to move new record into 2hr file.
        cmd = 'h5copy -i {newfile} -o {twohr} -s {dtstr} -d {dtstr}'
        cmd = cmd.format(newfile=tmp_file, twohr=out_file, dtstr=group_name)

        # Todo: improve call to subprocess.
        sp.call(cmd.split())
        os.remove(tmp_file)


def decompress_bz2(filename):
    basename = os.path.basename(filename)
    newfilepath = os.path.dirname(filename) + '/' + '.'.join(basename.split('.')[0:-1])  # all but bz2

    with open(newfilepath, 'wb') as new_file, bz2.BZ2File(filename, 'rb') as file:
        for data in iter(lambda: file.read(100 * 1024), b''):
            new_file.write(data)

    return newfilepath


def compress_bz2(filename):
    bz2_filename = filename + '.bz2'

    with open(filename, 'rb') as file, bz2.BZ2File(bz2_filename, 'wb') as bz2_file:
        for data in iter(lambda: file.read(100 * 1024), b''):
            bz2_file.write(data)

    return bz2_filename


def ptrs2antiq(filename, fixed_data_dir=''):
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

    if os.path.basename(filename).split('.')[-1] in ['bz2', 'bzip2']:
        hdf5_file = decompress_bz2(filename)
        bzip2 = True
    else:
        hdf5_file = filename
        bzip2 = False

    out_file = os.path.basename(hdf5_file).split('.')
    out_file = '.'.join(out_file[0:5]) + '.antennas_iq.hdf5.site'

    if fixed_data_dir == '/':
        out_file = fixed_data_dir + out_file
    elif fixed_data_dir == '':
        out_file = fixed_data_dir + out_file
    else:
        out_file = fixed_data_dir + "/" + out_file

    update_file(hdf5_file, out_file)

    return out_file


if __name__ == '__main__':
    
    # Todo (Adam): Need to make this tool properly callable from cli and not require a list.txt of
    #              files made from batch_log.py. Although, this may be a one-off tool.
    
    log_file = 'antennas_iq_files.txt'
    #files = [log_file]
    files = batch_log.read_file(log_file)
    for file in files[633::]:
        #print(file)
        #continue
        #name = os.path.basename(file).split('.')
        #name = '.'.join(name[0:5]) + '.antennas_iq.hdf5.site2'
        path = os.path.dirname(file).split('/')
        path = '/'.join(path[0:-2]) + '/sas_2019_processed/' + path[-1] + '/'
        ptrs2antiq(file, path)



