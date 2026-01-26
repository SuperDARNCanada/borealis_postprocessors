# Copyright 2021 SuperDARN Canada, University of Saskatchewan

"""
This file contains functions for averaging rawacfs
"""
from collections import OrderedDict
import numpy as np
from postprocessors import BaseConvert
import h5py
import pydarnio
from postprocessors.core.antennas_iq_to_rawacf import AntennasIQ2Rawacf
import logging

postprocessing_logger = logging.getLogger('borealis_postprocessing')


class RawacfAvg(BaseConvert):
    """
    Class for conversion of Widebeam to normalscan for beam-broadening experiments. This class
    inherits from BaseConvert, which handles all functionality generic to postprocessing borealis files.
    #note array type input does not work

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
        collected_indices = []  # list of what record (as an index number) each item in collected_timestamps belongs too

        with h5py.File(self.infile, 'r') as infile:
            all_records = sorted(list(infile.keys()))
            for i, rec in enumerate(all_records):
                inner_keys = list(infile[rec].keys())
                if "sqn_timestamps" in inner_keys:
                    collected_timestamps += list(infile[rec]['sqn_timestamps'][()])
                    collected_indices += [i]*len(infile[rec]['sqn_timestamps'][()])


        record_list = []  # A list of tuples that indicate the first and last records to grab for one process record call
        start = 0  # counter

        # loop over the timestamps until no more full averaging periods are available
        while start < len(collected_timestamps):
            first_tstamp = collected_timestamps[start]  # First sqn_timestamp of the average period
            idx_of_first_record = collected_indices[start]  # The corresponding index

            time_end = first_tstamp + avg_dur  # The end of the avg_period
            diff = np.abs(time_end - np.array(collected_timestamps))
            end = np.argmin(diff)  # The index in collected_timestamps for the closest timestamp to time_end
            last_tstamp = collected_timestamps[end]  # Last sqn_timestamp of the average period
            idx_of_last_record = collected_indices[end]  # The corresponding index
            num_sqn = end - start + 1  # Number of sequences to expect
            record_list.append((idx_of_first_record, idx_of_last_record, [first_tstamp, last_tstamp, num_sqn]))
            start = end + 1
        super().process_file(record_list = record_list, num_processes=1, **kwargs)

    @staticmethod
    def process_record(record: OrderedDict, extra_records, sqn_indices, **kwargs) -> OrderedDict:
        """
        Takes a set of records from a rawacf file and averages for a specified averaging_duration.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing antennas_iq data and metadata
        extra_records: list of OrderedDicts
            should be a list of OrderedDicts, which are the records to average.
        idxer: tuple
            Index of first record and index of last record to process at a time. Followed by list of the first sqn_tstamp, last sqn_tstamp and num_seqs for this average
        kwargs:
            Supported key: 'previous record'
            'previous_records' Previous records that may also be included in the average.
        Returns
        -------
        record: OrderedDict
            hdf5 record
        """

        # Combine the record lists into one
        total_records = [record] + extra_records

        end_points = idxer[2]  # Grab the value in end_points for this averaging period
        flag0 = False
        flag1 = False


        # Find the index in total records that corresponds to the end point values
        for i, rec in enumerate(total_records):
            # Check if the end_point is in this record
            temp_trunc_ind0 = np.where(rec['sqn_timestamps'] == end_points[0])
            temp_trunc_ind1 = np.where(rec['sqn_timestamps'] == end_points[1])
            # If it is flag this record index
            if temp_trunc_ind0[0].size != 0:
                trunc_ind0 = i
                flag0 = True
            if temp_trunc_ind1[0].size != 0:
                trunc_ind1 = i
                flag1 = True
            # if both flags raised -> exit
            if flag0 and flag1:
                break
        total_records = total_records[trunc_ind0:(trunc_ind1+1)]  # only keep the records we want
        total_seq = end_points[2]  # Total number of sequences

        if "data_dimensions" in list(record.keys()):
            record['data_dimensions'][1] = total_seq

        elif "correlation_dimensions" in list(record.keys()):
            record['correlation_dimensions'][1] = total_seq

        if 'data' in list(record.keys()):
            data_str = 'data'
        else:
            data_str = 'antennas_iq_data'



        # Initialize final record variables
        sqn_timestamps = list()
        int_time = 0
        noise_at_freq = list()
        gps_to_system_time_diff = list()

        data = np.array([])

        for i, rec in enumerate(total_records):
            rec_timestamps = np.array(list(map(float, rec['sqn_timestamps'])))  # sqn_timestamps for this record
            seq = np.where( (float(end_points[0])<=rec_timestamps) & (rec_timestamps<=float(end_points[1])) )[0]  # indices for sequences we keep

            # update the final record values
            sqn_timestamps.extend(list(rec['sqn_timestamps'][seq]))

            if 'noise_at_freq' in list(record.keys()):
                noise_at_freq.extend(rec['noise_at_freq'][seq])

            gps_to_system_time_diff.extend([rec['gps_to_system_time_diff']])

            avg_sqn= [rec_timestamps[i]-rec_timestamps[i-1] for i in range(1, len(rec_timestamps))]
            average_sqn= sum(avg_sqn)/len(avg_sqn)

            if (np.max(seq) +1) < len(rec_timestamps):
                int_time += rec_timestamps[np.max(seq) + 1] - rec_timestamps[seq][0]
            else:
                int_time += rec_timestamps[seq][-1] - rec_timestamps[seq][0] + average_sqn

            if i ==0:
                data = rec[data_str][:, seq, :]
            else:
                data = np.concatenate((data, rec[data_str][:, seq, :]), axis=1)

        record['int_time'] = np.float32(int_time)
        record['num_sequences'] = total_seq
        record['sqn_timestamps'] = sqn_timestamps
        if 'noise_at_freq' in list(record.keys()):
            record['noise_at_freq'] = noise_at_freq
        record[data_str] = data

        # Convert the AntennasIQ record to Rawacf
        toConvert = a2raw.AntennasIQ2Rawacf
        record = toConvert.process_record(record, **kwargs)

        record['main_acfs'] = np.complex64(record['main_acfs'])
        record['intf_acfs'] = np.complex64(record['intf_acfs'])
        record['xcfs'] = np.complex64(record['xcfs'])
        return record

