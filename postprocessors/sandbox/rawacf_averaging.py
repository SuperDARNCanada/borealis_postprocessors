# Copyright 2021 SuperDARN Canada, University of Saskatchewan

"""
This file contains functions for averaging rawacfs starting from an antennas_iq file
"""
from collections import OrderedDict
import numpy as np
from postprocessors import BaseConvert
import h5py
from postprocessors.core.antennas_iq_to_rawacf import AntennasIQ2Rawacf
from postprocessors import AntennasIQ2Bfiq
import logging

postprocessing_logger = logging.getLogger('borealis_postprocessing')


class RawacfAvg(BaseConvert):
    """
    Class for averaging rawacf's using inputted averaging duration from an "antennas_iq" File. This class
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
            Borealis structure of input file. Structure must be 'site'.
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

        collected_timestamps = []  # Total list sqn_timestamps across all records
        collected_indices = []  # list of what record (as index number) each item in collected_timestamps belongs too
                                # i.e first record -> idx of 0, second record -> idx of 1...
                                # -> [0, 0, ..., 0, 1, 1, ..., 1, 2, ..., 2,...]

        #  Fill in collected_timestamps and collected_indices
        with h5py.File(self.infile, 'r') as infile:
            all_records = sorted(list(infile.keys()))
            for i, rec in enumerate(all_records):
                inner_keys = list(infile[rec].keys())
                if "sqn_timestamps" in inner_keys:
                    collected_timestamps += list(infile[rec]['sqn_timestamps'][()])
                    collected_indices += [i]*len(infile[rec]['sqn_timestamps'][()])
            collected_indices = np.array(collected_indices) #to enable searching for sqn timestamp idx

        #  record_list: A list of tuples that indicate first and last records to grab for one process record call
        record_list = []  # i.e [(0,2), (2, 3)...], first process_record() will process records 0, 1 and 2

        sqn_indices = []  # additional information such as starting, ending ind, avg_dur and # of seq to avg

        start = 0 #The index in collected_timestamps for the first timestamp
        while start < len(collected_timestamps):
            #  Find the first timestamp, its rec idx then find the last timestamps rec idx
            first_tstamp = collected_timestamps[start]  # in first loop this is the first value of collected_timestamps
            idx_of_first_record = collected_indices[start]  # in first loop this is value will be 0
            time_end = first_tstamp + avg_dur

            diff = np.abs(time_end - np.array(collected_timestamps))
            end = np.argmin(diff)  # The index in collected_timestamps for the closest timestamp to time_end
            idx_of_last_record = collected_indices[end]

            num_sqn = end - start + 1  # Number of sequences to expect

            #  The indices of region corresponding to the first record/idx_of_first_record
            where_n_fir_rec = np.where(collected_indices[()]==idx_of_first_record)[0]
            #  To find the index that first_tstamp would appear in the first records' rec['sqn_timestamp']
            #  search for where in where_n_fir_rec is equal to start
            first_idx_tstmp = np.where(where_n_fir_rec == start)[0][0] #indicates the indice of starting sqn_timestamp

            #  The indices of region corresponding to the last record/idx_of_last_record
            where_n_las_rec = np.where(collected_indices == idx_of_last_record)[0]
            #  To find the index that last_tstamp would appear in the last records' rec['sqn_timestamp']
            #  search for where in where_n_las_rec is equal to end.
            #  NOTE: that in process_record the two records are concatenated, so we want the index w.r.t end of array
            #        This means negative indexing. To do this we should subtract by the length of last records'
            #        rec['sqn_timestamp'] which is the same as length of where_n_las_rec
            #        This indicates the indice of ending sqn_timestamp in reference to end

            last_idx_tstmp = np.where(where_n_las_rec == end)[0][0] -len(where_n_las_rec)

            sqn_indices.append([first_idx_tstmp, last_idx_tstmp,num_sqn, avg_dur])

            record_list.append((idx_of_first_record, idx_of_last_record))
            start = end + 1
        super().process_file(record_list = record_list, num_processes=1, sqn_indices = sqn_indices, **kwargs)

    @staticmethod
    def process_record(record: OrderedDict, extra_records, rec_indices, sqn_indices, **kwargs) -> OrderedDict:
        """
        Takes a set of records from a rawacf file and averages for a specified averaging_duration.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing antennas_iq data and metadata
        extra_records: list of OrderedDicts
            Additional records to average.
        rec_indices: int
            Index of what record or record set is being processed
        sqn_indices: list
            Index of first sequence, index of last sequence,
            averaging duration, and number of sequences to include in averaging.
            Indices are into sqn_timestamps when record and extra_records are flattened together.
        Returns
        -------
        record: OrderedDict
            hdf5 record
        """

        #  Combine the record lists into one
        total_records = [record] + extra_records

        #  Grab the values from sqn_indices for this averaging period
        start = sqn_indices[rec_indices][0]
        end = sqn_indices[rec_indices][1] + 1
        total_seq = sqn_indices[rec_indices][2]  # Total number of sequences
        avg_dur = sqn_indices[rec_indices][3]

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
                data = rec[data_str]
            else:
                data= np.concatenate((data, rec[data_str]), axis=1)

        #  Grab the sequences wanted
        if (end==0):  # At the end of array
            sqn_timestamps = np.array(sqn_timestamps[start:])
            noise_at_freq = np.array(noise_at_freq[start:])
            data = np.array(data[:, start:, :])
        else:  # Only want part of the array
            sqn_timestamps = np.array(sqn_timestamps[start:end])
            noise_at_freq = np.array(noise_at_freq[start:end])
            data = np.array(data[:, start:end, :])

        #  Update record values
        sqn_timestamps = np.array(sqn_timestamps)
        record['int_time'] = np.float32(avg_dur)
        record['num_sequences'] = total_seq
        record['sqn_timestamps'] = sqn_timestamps
        record['gps_to_system_time_diff'] = np.max(gps_to_system_time_diff)

        if 'noise_at_freq' in list(record.keys()):
            record['noise_at_freq'] = noise_at_freq
        record[data_str] = data

        #  Convert the AntennasIQ record to Rawacf
        record = AntennasIQ2Rawacf.process_record(record, **kwargs)
        return record
    @classmethod
    def _update_metadata(cls, record: OrderedDict, metadata: h5py.Group, **kwargs):
        """
        Adds in metadata fields required for this file type.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing one averaging period worth of data and metadata
        metadata: h5py.Group
            metadata group in the output file

        """
        averaging_method = kwargs.get("averaging_method", "mean")
        dset = metadata.create_dataset("averaging_method", data=averaging_method)
        dset.attrs["description"] = "Averaging method, e.g. mean, median"