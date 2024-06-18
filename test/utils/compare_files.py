# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains functions for comparing the contents of two HDF5 files.
"""
import h5py
import numpy as np


def compare_files(file1, file2):
    group1 = h5py.File(file1, 'r')
    group2 = h5py.File(file2, 'r')

    def compare_dictionaries(dict1, dict2, prefix):
        compare_string = ""
        keys1 = set(dict1.keys())
        keys2 = set(dict2.keys())
        uniq1 = keys1.difference(keys2)
        uniq2 = keys2.difference(keys1)
        shared = keys1.intersection(keys2)

        attrs1 = set(dict1.attrs.keys())
        attrs2 = set(dict2.attrs.keys())
        uniq_attrs1 = attrs1.difference(attrs2)
        uniq_attrs2 = attrs2.difference(attrs1)
        shared_attrs = attrs1.intersection(attrs2)

        try:
            assert len(uniq1) == len(uniq2) == 0
        except AssertionError:
            compare_string += prefix + f'Unique keys in {dict1}: {uniq1}\n' \
                            + prefix + f'Unique keys in {dict2}: {uniq2}\n'

        for key in shared:
            entry1 = dict1[key]
            entry2 = dict2[key]

            # Check that entries are the same type between files
            if type(entry1) != type(entry2):
                compare_string += prefix + f'Mismatched types for key {key}'
            # If they are dictionaries, recurse
            elif type(entry1) == h5py.Group:
                compare_dictionaries(entry1, entry2, prefix + '\t')
            # Otherwise, they must be lists or values, and so can be compared directly
            else:
                # Compare floating-point values differently
                if 'float' in str(entry1.dtype) or 'complex' in str(entry1.dtype):
                    if entry1.shape != entry2.shape:
                        entry1 = entry1[()].reshape(entry2.shape)

                    if not np.allclose(entry1, entry2, equal_nan=True):
                        compare_string += prefix + f"/{key}:\n" \
                                                   f"\t{entry1}\n" \
                                                   f"\t{entry2}\n" \
                                                   f"\tDifference: " \
                                                   f"{np.nanmax(np.abs((entry1[:] - entry2[:])/entry1[:]))}\n"
                # Comparing non-floating-point values
                elif not np.array_equal(entry1, entry2):
                    compare_string += prefix + f"/{key}:\n" \
                                               f"\t{entry1}\n" \
                                               f"\t{entry2}\n"

        return compare_string

    print(f"Comparing {file1} with {file2}:")
    print(compare_dictionaries(group1, group2, ""))
