import deepdish as dd
import numpy as np


def compare_files(file1, file2):
    group1 = dd.io.load(file1)
    group2 = dd.io.load(file2)

    def compare_dictionaries(dict1, dict2, prefix):
        compare_string = ""
        keys1 = set(dict1.keys())
        keys2 = set(dict2.keys())
        uniq1 = keys1.difference(keys2)
        uniq2 = keys2.difference(keys1)
        shared = keys1.intersection(keys2)

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
            elif type(entry1) == dict:
                compare_dictionaries(entry1, entry2, prefix + '\t')
            # Otherwise, they must be lists or values, and so can be compared directly
            else:
                equal = True

                # Compare floating-point values differently
                if 'float' in str(entry1.dtype) or 'complex' in str(entry1.dtype):
                    if not np.allclose(entry1, entry2, equal_nan=True):
                        equal = False
                # Comparing non-floating-point values
                elif not np.array_equal(entry1, entry2):
                    equal = False

                if not equal:
                    compare_string += prefix + f"/{key}\n"

        return compare_string

    print(f"Comparing {file1} with {file2}:")
    print(compare_dictionaries(group1, group2, ""))
