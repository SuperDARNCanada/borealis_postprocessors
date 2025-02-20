#!/bin/python3
# Testing antennas_iq to rawacf conversion
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
    site_outfile = 'test_antiq2rawacf.site'
    array_outfile = 'test_antiq2rawacf.array'

    compare_site_file = '{}/rawacf.site'
    compare_array_file = '{}/rawacf.array'

    for version in versions:
        print(version)
        if os.path.isfile(site_infile.format(version)):
            # convert site -> site
            if os.path.isfile(compare_site_file.format(version)):
                pp.ConvertFile(site_infile.format(version), site_outfile, 'antennas_iq', 'rawacf', 'site', 'site')
                compare_files(compare_site_file.format(version), site_outfile)
                os.remove(site_outfile)

            if int(version[1]) == 0:
                # convert site -> array:
                pp.ConvertFile(site_infile.format(version), array_outfile, 'antennas_iq', 'rawacf', 'site', 'array')
                compare_files(compare_array_file.format(version), array_outfile)
                os.remove(array_outfile)

        if os.path.isfile(array_infile.format(version)):
            # convert array -> site
            pp.ConvertFile(array_infile.format(version), site_outfile, 'antennas_iq', 'rawacf', 'array', 'site')
            compare_files(compare_site_file.format(version), site_outfile)
            os.remove(site_outfile)

            # convert array -> array
            pp.ConvertFile(array_infile.format(version), array_outfile, 'antennas_iq', 'rawacf', 'array', 'array')
            compare_files(compare_array_file.format(version), array_outfile)
            os.remove(array_outfile)
