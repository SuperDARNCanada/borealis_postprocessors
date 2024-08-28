"""
This file contains a class to modify the beam_azms metadata of antennas_iq files and reprocess them to rawacf.
"""

from collections import OrderedDict
import numpy as np
from postprocessors import AntennasIQ2Rawacf
from postprocessors.core.antennas_iq_to_bfiq import radar_dict


class UpdateBeamDirs(AntennasIQ2Rawacf):
    """
    Class for conversion of Borealis antennas_iq files into rawacf files with new beam directions.

    Attributes
    ----------
    infile: str
        The filename of the input antennas_iq file.
    outfile: str
        The file name of output file
    infile_structure: str
        The structure of the file. Structures include:
        'array'
        'site'
    outfile_structure: str
        The desired structure of the output file. Same structures as above, plus 'dmap'.
    """

    old_spacing = radar_dict["sas"]["main_antenna_spacing"]
    new_spacing = radar_dict["wal"]["main_antenna_spacing"]

    def __init__(
        self,
        infile: str,
        outfile: str,
        infile_structure: str,
        outfile_structure: str,
        **kwargs,
    ):
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
        super().__init__(infile, outfile, infile_structure, outfile_structure)

    def process_record(cls, record: OrderedDict, **kwargs) -> OrderedDict:
        """Update the beam_azms field, then process normally"""

        old_azms = record["beam_azms"]
        if isinstance(old_azms, list):
            new_azms = [
                np.rad2deg(
                    np.arcsin(
                        cls.old_spacing
                        * np.sin(np.deg2rad(theta))
                        / cls.new_spacing
                    )
                )
                for theta in old_azms
            ]
        else:
            new_azms = np.rad2deg(
                np.arcsin(
                    cls.old_spacing * np.sin(np.deg2rad(old_azms)) / cls.new_spacing
                )
            )

        record["beam_azms"] = new_azms
        record = super().process_record(record, **kwargs)

        return record
