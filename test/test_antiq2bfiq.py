#!/bin/python3
# Testing antennas_iq to bfiq conversion
import os

import conversion
from test.utils.compare_files import compare_files

array_infile = 'antennas_iq.hdf5'
site_infile = 'antennas_iq.hdf5.site'
site_outfile = 'test_antiq2bfiq.hdf5.site'
array_outfile = 'test_antiq2bfiq.hdf5'

compare_site_file = 'bfiq.hdf5.site'
compare_array_file = 'bfiq.hdf5'

# Convert from site file to both site and array files
conversion.ConvertFile(site_infile, site_outfile, 'antennas_iq', 'bfiq', 'site', 'site')
compare_files(compare_site_file, site_outfile)

conversion.ConvertFile(site_infile, array_outfile, 'antennas_iq', 'bfiq', 'site', 'array')
compare_files(compare_array_file, array_outfile)

# Remove the generated files
os.remove(site_outfile)
os.remove(array_outfile)

# Convert from array fil to both site and array files
conversion.ConvertFile(array_infile, site_outfile, 'antennas_iq', 'bfiq', 'array', 'site')
compare_files(compare_site_file, site_outfile)

conversion.ConvertFile(array_infile, array_outfile, 'antennas_iq', 'bfiq', 'array', 'array')
compare_files(compare_array_file, array_outfile)

# Remove the generate files
os.remove(site_outfile)
os.remove(array_outfile)
