#!/bin/python3
# Testing bfiq to rawacf conversion
import os

import postprocessors as pp
from test.utils.compare_files import compare_files

if __name__ == '__main__':
    array_infile = 'bfiq.hdf5'
    site_infile = 'bfiq.hdf5.site'
    site_outfile = 'test_bfiq2rawacf.hdf5.site'
    array_outfile = 'test_bfiq2rawacf.hdf5'

    compare_site_file = 'rawacf.alt.hdf5.site'
    compare_array_file = 'rawacf.alt.hdf5'

    # Convert from site file to both site and array files
    pp.ConvertFile(site_infile, site_outfile, 'bfiq', 'rawacf', 'site', 'site')
    compare_files(compare_site_file, site_outfile)

    pp.ConvertFile(site_infile, array_outfile, 'bfiq', 'rawacf', 'site', 'array')
    compare_files(compare_array_file, array_outfile)

    # Remove the generated files
    os.remove(site_outfile)
    os.remove(array_outfile)

    # Convert from array file to both site and array files
    pp.ConvertFile(array_infile, site_outfile, 'bfiq', 'rawacf', 'array', 'site')
    compare_files(compare_site_file, site_outfile)

    pp.ConvertFile(array_infile, array_outfile, 'bfiq', 'rawacf', 'array', 'array')
    compare_files(compare_array_file, array_outfile)

    # Remove the generate files
    os.remove(site_outfile)
    os.remove(array_outfile)
