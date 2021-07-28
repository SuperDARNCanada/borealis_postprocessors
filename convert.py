import argparse
import bz2
import os
import itertools
import subprocess as sp
import numpy as np
import warnings
import tables
import pydarnio

from bfiq_to_rawacf import BorealisBfiqToRawacfPostProcessor as bfiq2rawacf

warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
import deepdish as dd


import h5py


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


def batch_files(directory):
    """
    Create a list of all files and paths within a parent directory.
    """

    filepaths = []
    filenames = []

    for path, subdir, files in sorted(os.walk(directory)):
        if 'adamsenv' in path:
            continue
        else:
            for file in files:
                if ('.py' in file) or ('.txt' in file):
                    continue
                else:
                    filepaths.append(path)
                    filenames.append(file)

    return filepaths, filenames


class Convert():
    """
    This class takes any Borealis or SuperDARN data format and transfers it into any other type automatically.
    This only works to move data laterally or downstream, i.e. IQ -> rawacf but not the other way.

    Note
    ----
    This initially is designed to only change the old SAS files prior to September 9, 2019 to bfiq and rawacf.
    """

    def __init__(self):
        return


def update_file(filename, out_file, mode='antennas_iq'):
    recs = dd.io.load(filename)
    sorted_keys = sorted(list(recs.keys()))

    tmp_file = out_file + ".tmp"

    write_dict = {}

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

    for key_num, group_name in enumerate(sorted_keys):

        # APPLY CHANGE HERE
        # recs[group_name]['data_dimensions'][0] = 2
        if 'noise_at_freq' not in recs[group_name].keys():
            recs[group_name]['noise_at_freq'] = np.array([0.0] * int(recs[group_name]['num_sequences']),
                                                         dtype=np.float64)
            if key_num == 0:
                print('noise_at_freq added')
        if 'data_normalization_factor' not in recs[group_name].keys():
            recs[group_name]['data_normalization_factor'] = np.float64(9999999.999999996)
            if key_num == 0:
                print('data_normalization_factor added')
        if 'comment' in recs[group_name].keys():
            recs[group_name]['experiment_comment'] = recs[group_name]['comment']
            del recs[group_name]['comment']
            if key_num == 0:
                print('experiment_comment added')
        if 'slice_comment' not in recs[group_name].keys():
            recs[group_name]['slice_comment'] = np.unicode_('')
            if key_num == 0:
                print('slice_comment added')
        if 'experiment_string' in recs[group_name].keys():
            recs[group_name]['experiment_name'] = recs[group_name]['experiment_string']
            del recs[group_name]['experiment_string']
            if key_num == 0:
                print('experiment_name added')
        if 'num_slices' not in recs[group_name].keys():
            recs[group_name]['num_slices'] = np.int64(1)
            if key_num == 0:
                print('num_slices added')
        if 'timestamp_of_write' in recs[group_name].keys():
            del recs[group_name]['timestamp_of_write']
            if key_num == 0:
                print('timestamp_of_write removed')
        if not isinstance(recs[group_name]['experiment_id'], np.int64):
            recs[group_name]['experiment_id'] = np.int64(recs[group_name]['experiment_id'])
            if key_num == 0:
                print('experiment_id type changed')
        #{'experiment_name', 'noise_at_freq', 'data_normalization_factor', 'num_slices', 'pulse_phase_offset', 'num_ranges', 'experiment_comment', 'slice_comment'} 
        # Normalscan is default and has cpid=151 (experiment_id)
        if mode == 'bfiq':
            if 'pulse_phase_offset' not in recs[group_name].keys():
                recs[group_name]['pulse_phase_offset'] = np.array([], dtype=np.float32)
                if key_num == 0:
                    print('pulse_phase_offset added')
            if 'range_sep' not in recs[group_name].keys():
                recs[group_name]['range_sep'] = np.float32(44.96887)
                if key_num == 0:
                    print('range_sep added')
            if recs[group_name]['range_sep'] > 1000.0:
                recs[group_name]['range_sep'] = np.float32(recs[group_name]['range_sep'] / 1000.0)
                if key_num == 0:
                    print('range_sep changed due to bug')
            if 'num_ranges' not in recs[group_name].keys():
                recs[group_name]['num_ranges'] = np.uint32(75)
                if key_num == 0:
                    print('num_ranges added')
            if 'blanked_samples' not in recs[group_name].keys():
                recs[group_name]['blanked_samples'] = np.array([], dtype=np.int32)
                if key_num == 0:
                    print('blanked_samples added')
            if 'first_range' not in recs[group_name].keys():
                recs[group_name]['first_range'] = np.float32(0.0)
                if key_num == 0:
                    print('first_range added')
            lag_table = list(itertools.combinations(recs[group_name]['pulses'], 2))
            lag_table.append([recs[group_name]['pulses'][0], recs[group_name][
                'pulses'][0]])  # lag 0
            # sort by lag number
            lag_table = sorted(lag_table, key=lambda x: x[1] - x[0])
            lag_table.append([recs[group_name]['pulses'][-1], recs[group_name][
                'pulses'][-1]])  # alternate lag 0
            recs[group_name]['lags'] = np.array(lag_table, dtype=np.uint32)
            if key_num == 0:
                    print('lagtable generated due to unsorted')
            recs[group_name]['data_dimensions'][0] = 2  # only 2 antenna arrays not 3!

        # Sanity checks
        if 'main_acfs' in recs[group_name].keys():
            print(recs[group_name]['main_acfs'])
            print('Why is this in a bfiq!')
            exit()
        write_dict = {}
        write_dict[group_name] = convert_to_numpy(recs[group_name])
        dd.io.save(tmp_file, write_dict, compression=None)

        # use external h5copy utility to move new record into 2hr file.
        cmd = 'h5copy -i {newfile} -o {twohr} -s {dtstr} -d {dtstr}'
        cmd = cmd.format(newfile=tmp_file, twohr=out_file, dtstr=group_name)
        # print(cmd)
        # TODO(keith): improve call to subprocess.
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


def file_updater(filename, fixed_data_dir, mode='antennas_iq'):
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

    if fixed_data_dir[-1] == '/':
        out_file = fixed_data_dir + os.path.basename(hdf5_file)
    else:
        out_file = fixed_data_dir + "/" + os.path.basename(hdf5_file)

    update_file(hdf5_file, out_file, mode)

    if bzip2:
        # remove the input file from the directory because it was generated.
        os.remove(hdf5_file)
        # compress the updated file to bz2 if the input file was given as bz2.
        bz2_filename = compress_bz2(out_file)
        os.remove(out_file)
        out_file = bz2_filename

    return out_file


def batch_files_from_log(files_log):
    with open(files_log, 'r') as f:
        files = [line.rstrip() for line in f]
    f.close()
    return files
    
    
if __name__ == "__main__":
    # parser = script_parser()
    # args = parser.parse_args()
    # [YYYYmmDD].[HHMM].[SS].[station_id].[slice_id].[data_type].hdf5.[format].[compression]
    # 20191105.1400.02.sas.0.antennas_iq.hdf5.site
    

    files = batch_files_from_log('bfiq_files.txt')
    for f in files:
        print(f)
        df = os.path.dirname(f)
        df = df.split('/')
        df ='/'.join(df[0:3]) + '/sas_2019_processed/' + df[4] + '/'
        nf = file_updater(f, df, mode='bfiq')
        o = f.split('.')
        o = '.'.join(o[0:5]) + '.rawacf.hdf5.array'
        o = o.split('/')
        o[3] = 'sas_2019_rawacf'
        o = '/'.join(o[:])
        date_dir = o.split('/')
        date_dir = '/'.join(date_dir[:-1]) + '/'
        
        #data = h5py.File(f, 'r')
        #keys = list(data.keys())
        #print(data[f'{keys[0]}'].attrs['comment'])
        
        os.makedirs(date_dir, exist_ok=True)
        bfiq2rawacf(nf, o, 'site', 'array')
        print('write rawacf_file =', o)
        
        #{'experiment_name', 'noise_at_freq', 'data_normalization_factor', 'num_slices', 'pulse_phase_offset', 'num_ranges', 'experiment_comment', 'slice_comment'}

    exit()
    
    input_path = '/data/borealis_site_data/sas_2019_antennas_iq/20190511/'
    output_path = '/data/borealis_site_data/sas_2019_processed/'
    bfiq_path = '/data/borealis_site_data/sas_2019_bfiq/'
    rawacf_path = '/data/borealis_site_data/sas_2019_rawacf/'
    
    filepaths, filenames = batch_files(input_path)
    
    for i in range(len(filenames)):
        # Load files to be processed from /sas_2019_antennas_iq/
        input_file = filenames[i]
        input_path = filepaths[i]
        print('read input_file =', input_path + input_file)
        
        # Create the output directories
        date_dir = os.path.basename(os.path.dirname(input_path + '/' + input_file)) + '/'
        output_path += date_dir
        bfiq_path += date_dir
        rawacf_path += date_dir
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(bfiq_path, exist_ok=True)
        os.makedirs(rawacf_path, exist_ok=True)
    
        # Update the file save in /sas_2019_processed/
        fixed_file = file_updater(input_path + input_file, output_path)
        fixed_file = os.path.basename(fixed_file)
        print('write fixed_file =', fixed_file)

        # Convert to standard bfiq.hdf5.array non zipped and save in /sas_2019_bfiq/
        #test = '/data/borealis_site_data/sas_2019_processed/20190511/20190511.1000.02.sas.0.output_ptrs_iq.hdf5'
        reader = pydarnio.BorealisRead(output_path + fixed_file, 'antennas_iq', 'site')
        # reader = pydarnio.BorealisRead(test, 'antennas_iq', 'site')
        bfiq_file = fixed_file.split('.')
        bfiq_file = '.'.join(bfiq_file[0:5]) + '.bfiq.hdf5.site'
        #test = '/data/borealis_site_data/sas_2019_bfiq/20190511/20190511.1000.02.sas.0.bfiq.hdf5.site'
        pydarnio.BorealisWrite(bfiq_path + bfiq_file, reader.records, 'bfiq', 'site')
        #pydarnio.BorealisWrite(test, reader.records, 'bfiq', 'site')
        print('write bfiq_file =', bfiq_path + bfiq_file)

        # Process bfiq to rawacf and save in /sas_2019_rawacf/
        # This stage can be handled later once all the 2019 data is in bfiq.array standard
        rawacf_file = fixed_file.split('.')
        rawacf_file = '.'.join(rawacf_file[0:5]) + '.rawacf.hdf5.array'
        bfiq2rawacf(bfiq_path + bfiq_file, rawacf_path + rawacf_file, 'site', 'array')
        #bfiq2rawacf(test, rawacf_path + rawacf_file, 'site', 'array')
        print('write rawacf_file =', rawacf_path + rawacf_file)
        print('--next file')
        

        # Githash attrs() is not being found or added to entries after the first.
        # Site data is not mapping to array format properly.

        # TODO: Many files have missing fields that need to be determined through a labour intesive process
        #       or but analysis of the data itself :(
