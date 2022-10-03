# Borealis Postprocessors
A general purpose conversion and processing package for Borealis data files.

## Usage
This package can be used as a black-box for general conversion and processing of Borealis files. 
The file conversion.py contains the top-level functionality, and is the only file that the user needs to convert
files. The usage is as follows:

```
$ python3 conversion.py [-h] infile outfile infile_type outfile_type infile_structure outfile_structure [--averaging-method a]
```

Pass in the filename you wish to convert, the filename you wish to save as, and the types and structures of both.
The script will convert the input file into an output file of type "outfile_type" and structure "outfile_structure".
If the final type is rawacf, the averaging method may optionally be specified as well. 

If using this package from a python script, the usage is similar. Import `ConvertFile` into your script, then
initialize an instance of `ConvertFile` and all the conversion and processing will be done automatically. The usage for 
this method is as follows:

```python3
import postprocessors as pp

pp.ConvertFile(infile, outfile, infile_type, outfile_type, infile_structure, outfile_structure,
               averaging_method=averaging_method)
```

### Defining your own processing method
Following the template of any of the core processing chains (antennas_iq to bfiq, bfiq to rawacf, antennas_iq to rawacf),
it is fairly straightforward to define your own class for processing files. It is recommended you make this class
within the `/postprocessors/sandbox/` directory. Your class must have a few key components, following the template 
below:

```python3
from collections import OrderedDict
from typing import Union
from postprocessors import BaseConvert


class CustomProcessing(BaseConvert):
    """
    Custom class for processing SuperDARN Borealis files.
    """

    def __init__(self, infile: str, outfile: str, infile_type: str, outfile_type: str, infile_structure: str, 
                 outfile_structure: str, **kwargs):
        """
        Feel free to add more parameters to this, or hard-code in the file types and structures if you don't need 
        them to be variable.

        Parameters
        ----------
        infile: str
            Path to input file.
        outfile: str
            Path to output file.
        infile_type: str
            Borealis data type of input file. One of 'antennas_iq', 'bfiq', or 'rawacf'.
        outfile_type: str
            Borealis data type of output file. One of 'antennas_iq', 'bfiq', or 'rawacf'.
        infile_structure: str
            Borealis structure of input file. Either 'array' or 'site'.
        outfile_structure: str
            Borealis structure of output file. Either 'array', 'site', or 'dmap'.
        """
        super().__init__(infile, outfile, infile_type, outfile_type, infile_structure, outfile_structure)

        # You can pass extras parameters needed for process_record() as kwargs

        self.process_file(**kwargs)

    @staticmethod
    def process_record(record: OrderedDict, averaging_method: Union[None, str], **kwargs) -> OrderedDict:
        """
        Takes a record from a data file and processes it.
        This method is called from within self.process_file() above, and handles the processing of one record
        from a site-structured Borealis file. It is up to you to verify that this processing is handled correctly.

        Parameters
        ----------
        record: OrderedDict
            hdf5 record containing antennas_iq data and metadata
        averaging_method: Union[None, str]
            Method to use for averaging correlations across sequences. Acceptable methods are 'median' and 'mean'
        kwargs: dict
            Whatever you need to do your processing!
            
        Returns
        -------
        record: OrderedDict
            hdf5 record, with new fields required by output data type
        """
        # do your processing here
        return record
```

## Package Structure

### postprocessors
Contains the data processing chain, for all processing stages. 

#### core
The main classes in this module are BaseConvert, ProcessAntennasIQ2Bfiq, ProcessBfiq2Rawacf, and
ProcessAntennasIQ2Rawacf, and ConvertFile. These classes handle their respective processing between data levels. 
Additionally, BaseConvert handles restructuring between file structures, using pyDARNio.

#### exceptions
Contains all exceptions that the package will throw. 

#### sandbox
Extra processing classes for doing non-standard processing of files. This is the place to add new classes
for processing any non-standard experiments that you may run.

#### utils
Some handy utility functions for processing. Currently, the only functions are for generating filenames for downstream
data files.

### scripts
This directory houses scripts for calling the processing within the `postprocessors` package. 
If you plan on batch-processing some data, it is recommended you create your script here.

### test
This directory houses test scripts and data for verifying the core processing functions of this package.