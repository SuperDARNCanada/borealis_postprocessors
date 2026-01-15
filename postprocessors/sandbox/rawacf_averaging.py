# Copyright 2021 SuperDARN Canada, University of Saskatchewan

"""
This file contains functions for averaging rawacfs
"""
from collections import OrderedDict
from typing import Union
import numpy as np
from postprocessors import BaseConvert
import postprocessors
import os
import traceback
import h5py
from functools import partial
from multiprocessing import get_context
import pydarnio
import postprocessors.core.convert_base as cb
import postprocessors.core.restructure as rs
from postprocessors import conversion_exceptions
import datetime as dt

try:
    import cupy as xp
except ImportError:
    import numpy as xp

    cupy_available = False
else:
    cupy_available = True

import logging

postprocessing_logger = logging.getLogger('borealis_postprocessing')


class Rawacf_Avg(BaseConvert):
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

    Attributes
    ----------
    infile: str
        The filename of the input antennas_iq file.
    outfile: str
        The file name of output file
    infile_structure: str
        The write structure of the file. Structures include:
        'array'
        'site'
    outfile_structure: str
        The desired structure of the output file. Same structures as above, plus 'dmap'.
    """

    def __init__(self, infile: str, outfile: str, infile_structure: str, outfile_structure: str):
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
            'rawacf'
        outfile_type: str
            Desired type of output data file. Same types as above.
        infile_structure: str
            Borealis structure of input file. Either 'array' or 'site'.
        outfile_structure: str
            Borealis structure of output file. Either 'array', 'site', or 'dmap'.
        """
        super().__init__(infile, outfile, "rawacf", "rawacf", infile_structure, outfile_structure)

    def process_file(self, avg_dur: float = 3.7, **kwargs):
        """
        Takes a rawacf file and averages records from the file based on a given averaging_duration.

        Parameters
        ----------
        avg_dur: Float
            Averaging duration in seconds to be used.
        """

        collected_timestamps = [] #Total list sqn_timestamps
        all_records = [] #Total list of keys from all records
        collected_indices = [] #list of what record (as an index number) each item in collected_timestamps belongs too

        with h5py.File(self.infile, 'r') as infile:
            all_records += sorted(list(infile.keys()))
            for i, rec in enumerate(all_records):
                collected_timestamps += list(infile[rec]['sqn_timestamps'])
                collected_indices += [i]*len(infile[rec]['sqn_timestamps'])
        collected_timestamps =list(map(float, collected_timestamps))

        end_points = [] #Holds what the first sqn_timestamp and last sqn_timestamp for each averaging duration and num of sequences to expect
        record_list = [] #A list of tuples that indicate the first and last records to grab for one process record call
        start = 0 #counter

        #loop over the timestamps until no more full averaging periods are available
        while start < len(collected_timestamps):
            first = collected_timestamps[start] #First sqn_timestamp of the average period
            first_index = collected_indices[start] #The corresponding index

            time_end = first + avg_dur #The end of the avg_period
            diff = np.abs(time_end - np.array(collected_timestamps))
            end = np.argmin(diff) #The index in collected_timestamps for the closest timestamp to time_end
            if (end+1) >= len(collected_timestamps): #if there are no more full periods
                break
            else:
                last = collected_timestamps[end] #Last sqn_timestamp of the average period
                last_index = collected_indices[end] #The corresponding index
            num_seq = len(collected_timestamps[start:(end+1)]) #Number of sequences to expect
            record_list.append((first_index, last_index))
            start = end + 1
            end_points.append([first, last, num_seq])
        super().process_file(record_list = record_list, end_points = end_points, num_processes=1, **kwargs)

    @staticmethod
    def process_record(record: OrderedDict, averaging_method: Union[None, str] = 'mean', **kwargs) -> OrderedDict:
        """
        Takes a set of records from a rawacf file and averages for a specified averaging_duration.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing antennas_iq data and metadata
        averaging_method: Union[None, str]
            Method to use for averaging correlations across sequences. For this class, only 'mean' averaging is
            supported, as the median cannot be taken across multiple records after rawacf files have been made.
        kwargs:
            Supported key: 'extra_records'
            'extra_records' should be a list of OrderedDicts, which are the records to average.
            'previous_records' Previous records that may also be included in the average.
            'end_points' A list of lists, which hold three values:
                         The starting and ending times for each average period and number of sequences.
        Returns
        -------
        record: OrderedDict
            hdf5 record
        """
        if 'extra_records' not in kwargs:
            print("No extra records given.")
            return record
        if 'previous_records' not in kwargs:
            print("No previous_records given.")
            return record
        if 'end_points' not in kwargs:
            print("No end points given.")
            return record

        index = 0 #What averaging period we are in
        #Find what averaging period we are in
        for i, k in enumerate(kwargs['end_points']):
            if k== [0, 0, 0]:
                index = i + 1

        #Combine the record lists into one
        total_records = kwargs['previous_records'] + [record] + kwargs['extra_records']

        end_points = kwargs['end_points'][index] #Grab the value in end_points for this averaging period
        flag0 = False
        flag1 = False

        #Find the index in total records that corresponds to the end point values
        for i, rec in enumerate(total_records):
            # Check if the end_point is in this record
            temp_trunc_ind0 = np.where(rec['sqn_timestamps'] == end_points[0])
            temp_trunc_ind1 = np.where(rec['sqn_timestamps'] == end_points[1])
            #If it is flag this record index
            if temp_trunc_ind0[0].size != 0:
                trunc_ind0 = i
                flag0 = True
            if temp_trunc_ind1[0].size != 0:
                trunc_ind1 = i
                flag1 = True
            #if both flags raised -> exit
            if flag0 and flag1:
                break
        total_records = total_records[trunc_ind0:(trunc_ind1+1)] #only keep the records we want
        total_seq = end_points[2] #Total number of sequences

        # Initialize final record variables
        sqn_timestamps = list()
        int_time = 0
        noise_at_freq = list()
        main_acfs = record['main_acfs'] * 0
        intf_acfs = record['intf_acfs'] * 0
        xcfs = record['xcfs'] * 0

        for rec in total_records:
            num_sequences = rec['num_sequences']

            old_main_acfs = rec['main_acfs']*num_sequences
            old_intf_acfs = rec['intf_acfs']*num_sequences
            old_xcfs = rec['xcfs']*num_sequences

            #Since we are not averaging based on subsequent records we need the unaveraged correlations
            #To "pseudo undo" the averaging, a normal distribution was created to recreate the unaveraged correlations
            rng= np.random.default_rng()
            filter = np.abs(rng.normal(0, 0.1, size=num_sequences))
            filter = filter/sum(filter)
            un_avg_main_acf = np.array([old_main_acfs*weight for weight in filter])
            un_avg_intf_acf = np.array([old_intf_acfs*weight for weight in filter])
            un_avg_xcf = np.array([old_xcfs*weight for weight in filter])

            rec_timestamps = np.array(list(map(float, rec['sqn_timestamps']))) #sqn_timestamps for this record
            seq = np.where( (float(end_points[0])<=rec_timestamps) & (rec_timestamps<=float(end_points[1])) )[0] #indices for sequences we keep

            #update the final record values
            sqn_timestamps.extend(list(rec['sqn_timestamps'][seq]))
            noise_at_freq.extend(rec['noise_at_freq'][seq])
            avg_sqn= [rec_timestamps[i]-rec_timestamps[i-1] for i in range(1, len(rec_timestamps))]
            average_sqn= sum(avg_sqn)/len(avg_sqn)
            if (np.max(seq) +1) < len(rec_timestamps):
                int_time += rec_timestamps[np.max(seq) + 1] - rec_timestamps[seq][0]
            else:
                int_time += rec_timestamps[seq][-1] - rec_timestamps[seq][0] + average_sqn

            un_avg_main_acf = un_avg_main_acf[seq]
            un_avg_intf_acf = un_avg_intf_acf[seq]
            un_avg_xcf = un_avg_xcf[seq]
            #Sum the correlations
            avg_main_acf = np.einsum('ijkl->jkl', un_avg_main_acf)
            avg_intf_acf = np.einsum('ijkl->jkl', un_avg_intf_acf)
            avg_xcf = np.einsum('ijkl->jkl', un_avg_xcf)

            #update ACFS portion of record
            main_acfs += avg_main_acf
            intf_acfs += avg_intf_acf
            xcfs += avg_xcf

        main_acfs /= total_seq
        intf_acfs /= total_seq
        xcfs /= total_seq

        record['main_acfs'] = np.array(main_acfs, dtype=np.complex64)
        record['intf_acfs'] = np.array(intf_acfs, dtype=np.complex64)
        record['xcfs'] = np.array(xcfs, dtype=np.complex64)
        record['int_time'] = np.float32(int_time)
        record['num_sequences'] = total_seq
        record['sqn_timestamps'] = sqn_timestamps
        record['noise_at_freq'] = noise_at_freq

        # indicate that this period has been processed
        kwargs['end_points'][index][0] = 0
        kwargs['end_points'][index][1] = 0
        kwargs['end_points'][index][2] = 0

        return record

