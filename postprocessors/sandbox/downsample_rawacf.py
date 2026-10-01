# Copyright 2022 SuperDARN Canada, University of Saskatchewan

"""
This file contains functions for downsampling records of a rawacf file.
With two options, downsampling widebeam experiments back to normalscan or
keeping widebeam format and downsampling to 1 min integration time
"""
from collections import OrderedDict
from postprocessors import BaseConvert
from postprocessors.sandbox.widebeam_downsample2normalscan import Widebeam2NormalScan
from postprocessors.sandbox.rawacf_record_averaging import AverageMultipleRawacfRecords


class Downsample_Rawacf(BaseConvert):
    """
    Class for downsampling rawacf records. This class inherits from BaseConvert, which handles all
    functionality generic to postprocessing borealis files.

    See Also
    --------
    ConvertFile
    BaseConvert
    ProcessBfiq2Rawacf
    ProcessAntennasIQ2Rawacf

    Attributes
    ----------
    infile: str
        The filename of the input rawacf file.
    outfile: str
        The file name of output file
    infile_structure: str
        The write structure of the file. Structures include:
        'dmap'
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
        infile_structure: str
            Borealis structure of input file. Either 'array' or 'site'.
        outfile_structure: str
            Borealis structure of output file. Either 'array', 'site', or 'dmap'.
        """
        super().__init__(infile, outfile, 'rawacf', 'rawacf', infile_structure, outfile_structure)



    def process_file(self, dwn_smp_mode: str = "normal_scan", **kwargs):
        kwargs['dwn_smp_mode'] = dwn_smp_mode
        if dwn_smp_mode == "normal_scan":  # Widebeam to normal scan
            super().process_file(avg_num = 16, **kwargs)
        elif dwn_smp_mode == "1min":  # Keep widebeam but downsample 1 minute scans instead of int_time
            super().process_file(avg_num=16, same_stamp=False, **kwargs)
        else:
            print("Unknown dwn_smp_mode {}".format(dwn_smp_mode))

    @staticmethod
    def process_record(record: OrderedDict, **kwargs) -> OrderedDict:
        """ Depending on downsample mode process the hdf5 record accordingly.
            Then update the experiment comment
        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing rawacf data and metadata for beam 0
        Returns
        -------
        records: list[OrderedDict]
            list of 16 records, each downsampled according to dwn_smp_mode
        """
        dwn_smp_mode = kwargs.get('dwn_smp_mode', 'normal_scan')
        if dwn_smp_mode == "normal_scan":
           records = Widebeam2NormalScan.process_record(record, **kwargs)
        elif dwn_smp_mode == "1min":
            records = AverageMultipleRawacfRecords.process_record(record, **kwargs)
        else:
            print("Unknown dwn_smp_mode {}".format(dwn_smp_mode))
        if isinstance(records, list):
            for rec in records:
                rec['experiment_comment'] = rec['experiment_comment'] + f' downsampled: {dwn_smp_mode}'
        else:
            records['experiment_comment'] = records['experiment_comment'] + f' downsampled: {dwn_smp_mode}'

        return records

    @staticmethod
    def process_record_dmap(record: OrderedDict, **kwargs) -> OrderedDict:
        """ Depending on downsample mode process the dmap record accordingly.
            Then update the origin.command
        Parameters
        ----------
        record: list[OrderedDict]
            dmap list of records record containing rawacf data and metadata for int_time[0]
        Returns
        -------
        records: list[OrderedDict]
            list of 16 records, each downsampled according to dwn_smp_mode
        """
        dwn_smp_mode = kwargs.get('dwn_smp_mode', 'normal_scan')
        if dwn_smp_mode == "normal_scan":
           records = Widebeam2NormalScan.process_record_dmap(record, **kwargs)
        elif dwn_smp_mode == "1min":
            records = AverageMultipleRawacfRecords.process_record_dmap(record, **kwargs)
        else:
            print("Unknown dwn_smp_mode {}".format(dwn_smp_mode))
        for rec in records:
            rec['origin.command'] = rec['origin.command'] + f' downsampled: {dwn_smp_mode}'
        return records
