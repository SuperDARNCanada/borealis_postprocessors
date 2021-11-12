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
from conversion import ConvertFile
ConvertFile(infile, outfile, infile_type, outfile_type, infile_structure, outfile_structure,
            averaging_method=averaging_method)
```

## Package Structure

### data_processing
Contains the data processing chain, for all processing stages. 
The main classes in this module are BaseConvert, ProcessAntennasIQ2Bfiq, ProcessBfiq2Rawacf, and
ProcessAntennasIQ2Rawacf. These classes handle their respective processing between data levels. Additionally,
BaseConvert handles restructuring between file structures, using pyDARNio.

### exceptions
Contains all exceptions that the package will throw. 