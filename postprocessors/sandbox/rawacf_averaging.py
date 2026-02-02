# Copyright 2021 SuperDARN Canada, University of Saskatchewan

"""
This file contains functions for averaging rawacfs
"""
from collections import OrderedDict
import numpy as np
from postprocessors import BaseConvert
import h5py
from postprocessors.core.antennas_iq_to_rawacf import AntennasIQ2Rawacf
import logging

postprocessing_logger = logging.getLogger('borealis_postprocessing')


class RawacfAvg(BaseConvert):
    """
    Class for averaging rawacf's using users inputted averaging duration from an antennasIQ File. This class
    inherits from BaseConvert, which handles all functionality generic to postprocessing borealis files.

    See Also
    --------
    ConvertFile
    BaseConvert
    ProcessAntennasIQ2Bfiq
    ProcessBfiq2Rawacf
    ProcessAntennasIQ2Rawacf

    """

    def __init__(self, infile: str, outfile: str, infile_type: str, infile_structure: str, outfile_structure: str):
        """
        Initialize the attributes of the class.

        Parameters
        ----------
        infile: str
            Path to input file.
        outfile: str
            Path to output file.
        infile_type: str
            Type of data file. Types include:
            'antennas_iq, bfiq'
        infile_structure: str
            Borealis structure of input file. Either 'array' or 'site'.
        outfile_structure: str
            Borealis structure of output file. Either 'array', 'site', or 'dmap'.
        """
        if infile_structure not in ['site']:
            print("Invalid infile_structure argument -> Only site files are accepted")
            return
        else:
            super().__init__(infile, outfile, infile_type, "rawacf", infile_structure, outfile_structure)

    def process_file(self, avg_dur: float = 3.7, **kwargs):
        """
        Takes a rawacf file and averages records from the file based on a given `avg_dur` in seconds.

        Parameters
        ----------
        avg_dur: Float
            Averaging duration in seconds to be used.
        """

        collected_timestamps = []  # Total list sqn_timestamps
        collected_indices = []  # list of what record (as index number) each item in collected_timestamps belongs too

        with h5py.File(self.infile, 'r') as infile:
            all_records = sorted(list(infile.keys()))
            for i, rec in enumerate(all_records):
                inner_keys = list(infile[rec].keys())
                if "sqn_timestamps" in inner_keys:
                    collected_timestamps += list(infile[rec]['sqn_timestamps'][()])
                    collected_indices += [i]*len(infile[rec]['sqn_timestamps'][()])
            collected_indices = np.array(collected_indices)

        record_list = []  # A list of tuples that indicate first and last records to grab for one process record call
        add_avg_info = []
        start = 0
        while start < len(collected_timestamps):
            #  Find the first timestamp, its rec idx then find the last timestamps rec idx
            first_tstamp = collected_timestamps[start]  # First sqn_timestamp of the average period
            idx_of_first_record = collected_indices[start]  # The corresponding index
            time_end = first_tstamp + avg_dur  # The end of the avg_period

            diff = np.abs(time_end - np.array(collected_timestamps))
            end = np.argmin(diff)  # The index in collected_timestamps for the closest timestamp to time_end
            idx_of_last_record = collected_indices[end]  # The corresponding index

            num_sqn = end - start + 1  # Number of sequences to expect

            # The indices of region corresponding to the first record
            where_col_idx = np.where(collected_indices==idx_of_first_record)[0]
            #  Match the indice corresponding to start
            first_idx_tstmp = np.where(where_col_idx == start)[0][0] #indicates the indice of starting sqn_timestamp

            # The indices of region corresponding to the last record
            where_col_idx = np.where(collected_indices == idx_of_last_record)[0]
            #  Match the indice corresponding to end
            #  This indicates the indice of ending sqn_timestamp in reference to end
            last_idx_tstmp = np.where(where_col_idx == end)[0][0] -len(np.where(collected_indices==idx_of_last_record)[0])
            add_avg_info.append([first_idx_tstmp, last_idx_tstmp,num_sqn, avg_dur])

            record_list.append((idx_of_first_record, idx_of_last_record))
            start = end + 1
        super().process_file(record_list = record_list, num_processes=1, add_avg_info = add_avg_info, **kwargs)

    @staticmethod
    def process_record(record: OrderedDict, extra_records, sqn_indices, add_avg_info, **kwargs) -> OrderedDict:
        """
        Takes a set of records from a rawacf file and averages for a specified averaging_duration.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing antennas_iq data and metadata
        extra_records: list of OrderedDicts
            Additional records to average.
        sqn_indices: tuple
            Index of first sequence, index of last sequence, and number of sequences to include in averaging.
            Indices are into sqn_timestamps when record and extra_records are flattened together.
        add_avg_info: list
            Holds additional average info:
            Index of first sequence, Index of last sequence, number of sequences to average, and averaging_duration.
        Returns
        -------
        record: OrderedDict
            hdf5 record
        """

        # Combine the record lists into one
        total_records = [record] + extra_records

        end_points = add_avg_info[sqn_indices]  # Grab the value in end_points for this averaging period

        total_seq = end_points[2]  # Total number of sequences

        if "data_dimensions" in list(record.keys()):
            record['data_dimensions'][1] = total_seq

        elif "correlation_dimensions" in list(record.keys()):
            record['correlation_dimensions'][1] = total_seq

        if 'data' in list(record.keys()):
            data_str = 'data'
        else:
            data_str = 'antennas_iq_data'


        #  Initialize final record variables
        sqn_timestamps = list()
        noise_at_freq = list()
        gps_to_system_time_diff = list()

        for i, rec in enumerate(total_records):
            #  Concatenate all sequences next to each other
            sqn_timestamps.extend(list(rec['sqn_timestamps'][()]))
            gps_to_system_time_diff.extend([rec['gps_to_system_time_diff']])
            if 'noise_at_freq' in list(record.keys()):
                noise_at_freq.extend(rec['noise_at_freq'][()])
            if i == 0:
                data = rec[data_str][:, :, :]
            else:
                data = np.concatenate((data, rec[data_str][:, :, :]), axis=1)

        #  Grab the sequences wanted
        if ((end_points[1]+1)==0):
            sqn_timestamps = np.array(sqn_timestamps[end_points[0]:])
            noise_at_freq = np.array(noise_at_freq[end_points[0]:])
            data = np.array(data[:, end_points[0]:, :])
        else:
            end = end_points[1] + 1
            sqn_timestamps = np.array(sqn_timestamps[end_points[0]:end])
            noise_at_freq = np.array(noise_at_freq[end_points[0]:end])
            data = np.array(data[:, end_points[0]:end, :])

        sqn_timestamps = np.array(sqn_timestamps)
        record['int_time'] = np.float32(end_points[3])
        record['num_sequences'] = total_seq
        record['sqn_timestamps'] = sqn_timestamps
        record['gps_to_system_time_diff'] = np.max(gps_to_system_time_diff)

        if 'noise_at_freq' in list(record.keys()):
            record['noise_at_freq'] = noise_at_freq
        record[data_str] = data

        # Convert the AntennasIQ record to Rawacf
        record = AntennasIQ2Rawacf.process_record(record, **kwargs)
        return record

