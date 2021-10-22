import os
import h5py
import csv


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


def write_file(files_list, log_file):
    f = open(log_file, 'w+')
    for i in range(len(files_list)):
        f.write(f'{files_list[i]}\n')
    f.close()
    return


def read_file(log_file):
    files = []
    f = open(log_file, 'r')
    files = f.readlines()
    files = [x.strip() for x in files]
    return files


def check_keys(log_file):
    files = read_file(log_file)
    n = len(files)
    flag = 0
    for f in files:
        data = h5py.File(f, 'r')
        groups = list(data.keys())
        keys = list(data[groups[0]].keys())
        if flag == 0:
            old_keys = keys.copy()
            flag += 1
        if keys != old_keys:
            print(f, keys)
        else:
            print(f'no changes: {flag}/{n}')
            flag += 1
        data.close()

    return


def write_csv(log_file):
    files = read_file(log_file)
    csv_file = log_file.split('.')
    csv_file = csv_file[0] + '.csv'
    g = open(csv_file, 'w')
    writer = csv.writer(g)
    for f in files:
        data = h5py.File(f, 'r')
        groups = list(data.keys())
        keys = list(data[groups[0]].keys())
        attrs = list(data[groups[0]].attrs.keys())
        row = [f] + keys + attrs
        print(row)
        writer.writerow(row)

    g.close()
    return


if __name__ == "__main__":
    #write_csv('output_ptrs_files.txt')
    #check_keys('output_ptrs_files.txt')
    #exit()

    #directory = '/data/borealis_site_data/sas_2019_antennas_iq/'
    directory = '/data/borealis_site_data/sas_2019_processed/'

    filepaths, filenames = batch_files(directory)
    
    antennas_iq_files = []
    bfiq_files = []
    rawacf_files = []
    output_ptrs_files = []
    other_files = []
    
    for i in range(len(filenames)):
        if 'antennas_iq' in filenames[i]:
            antennas_iq_files.append(filepaths[i] + '/' + filenames[i])
        elif 'bfiq' in filenames[i]:
            bfiq_files.append(filepaths[i] + '/' + filenames[i])
        elif 'rawacf' in filenames[i]:
            rawacf_files.append(filepaths[i] + '/' + filenames[i])
        elif 'output_ptrs' in filenames[i]:
            output_ptrs_files.append(filepaths[i] + '/' + filenames[i])
        else:
            other_files.append(filepaths[i] + '/' + filenames[i])

    #write_file(antennas_iq_files, 'antennas_iq_files.txt')
    write_file(bfiq_files, 'processed_bfiq_files.txt')
    #write_file(rawacf_files, 'rawacf_files.txt')
    #write_file(output_ptrs_files, 'output_ptrs_files.txt')
    #write_file(other_files, 'other_files.txt')
