import os


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

    
if __name__ == "__main__":
    directory = '/data/borealis_site_data/sas_2019_antennas_iq/'
    
    filepaths, filenames = batch_files(directory)
    
    antennas_iq_files = []
    bfiq_files = []
    rawacf_files = []
    other_files = []
    
    for i in range(len(filenames)):
        if 'antennas_iq' in filenames[i]:
            antennas_iq_files.append(filepaths[i] + '/' + filenames[i])
        elif 'bfiq' in filenames[i]:
            bfiq_files.append(filepaths[i] + '/' + filenames[i])
        elif 'rawacf' in filenames[i]:
            rawacf_files.append(filepaths[i] + '/' + filenames[i])
        else:
            other_files.append(filepaths[i] + '/' + filenames[i])

    write_file(antennas_iq_files, 'antennas_iq_files.txt')
    write_file(bfiq_files, 'bfiq_files.txt')
    write_file(rawacf_files, 'rawacf_files.txt')
    write_file(other_files, 'other_files.txt')
