#!/bin/python3
# Testing bfiq to rawacf conversion
import os

import postprocessors as pp
from test.utils.compare_files import compare_files

if __name__ == '__main__':

    versions = [
        "v0.5",
        "v0.7",
        "v1.0"
    ]

    array_infile = '{}/bfiq.array'
    site_infile = '{}/bfiq.site'
    site_outfile = 'test_bfiq2rawacf.site'
    array_outfile = 'test_bfiq2rawacf.array'

    compare_site_file = '{}/rawacf.site'
    compare_array_file = '{}/rawacf.array'

    for version in versions:
        print(version)

        if os.path.isfile(site_infile.format(version)):
            # site -> site
            pp.ConvertFile(site_infile.format(version), site_outfile, 'bfiq', 'rawacf', 'site', 'site')
            compare_files(compare_site_file.format(version), site_outfile)
            os.remove(site_outfile.format(version))

            if int(version[1]) == 0:
                # site -> array
                pp.ConvertFile(site_infile.format(version), array_outfile, 'bfiq', 'rawacf', 'site', 'array')
                compare_files(compare_array_file.format(version), array_outfile)
                os.remove(array_outfile.format(version))

        if os.path.isfile(array_infile.format(version)):
            # array -> site
            pp.ConvertFile(array_infile.format(version), site_outfile, 'bfiq', 'rawacf', 'array', 'site')
            compare_files(compare_site_file.format(version), site_outfile)
            os.remove(site_outfile.format(version))

            # array -> array
            pp.ConvertFile(array_infile.format(version), array_outfile, 'bfiq', 'rawacf', 'array', 'array')
            compare_files(compare_array_file.format(version), array_outfile)
            os.remove(array_outfile.format(version))
