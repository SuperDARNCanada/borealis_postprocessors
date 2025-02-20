#!/bin/python3
# Testing antennas_iq to bfiq conversion
import os

import postprocessors as pp
from test.utils.compare_files import compare_files

if __name__ == '__main__':

    versions = [
        "v0.5",
        "v0.7",
        "v1.0"
    ]

    array_infile = '{}/antennas_iq.array'
    site_infile = '{}/antennas_iq.site'
    site_outfile = 'test_antiq2bfiq.site'
    array_outfile = 'test_antiq2bfiq.array'

    compare_site_file = '{}/bfiq.site'
    compare_array_file = '{}/bfiq.array'

    for version in versions:
        print(version)

        if os.path.isfile(site_infile.format(version)):
            # convert site -> site
            pp.ConvertFile(site_infile.format(version), site_outfile, 'antennas_iq', 'bfiq', 'site', 'site')
            compare_files(compare_site_file.format(version), site_outfile)
            os.remove(site_outfile)

            if int(version[1]) == 0:
                # convert site -> array
                pp.ConvertFile(site_infile.format(version), array_outfile, 'antennas_iq', 'bfiq', 'site', 'array')
                compare_files(compare_array_file.format(version), array_outfile)
                os.remove(array_outfile)

        if os.path.isfile(array_infile.format(version)) and int(version[1]) == 0:
            # convert array -> site
            pp.ConvertFile(array_infile.format(version), site_outfile, 'antennas_iq', 'bfiq', 'array', 'site')
            compare_files(compare_site_file.format(version), site_outfile)
            os.remove(site_outfile)

            # convert array -> array
            pp.ConvertFile(array_infile.format(version), array_outfile, 'antennas_iq', 'bfiq', 'array', 'array')
            compare_files(compare_array_file.format(version), array_outfile)
            os.remove(array_outfile)
