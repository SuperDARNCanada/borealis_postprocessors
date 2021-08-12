import h5py
import argparse

parser = argparse.ArgumentParser(description='Dump all keys and attributes in a HDF5 file.')
parser.add_argument('-f', dest='filepath', help='HDF5 file and path')
args = parser.parse_args()

f = h5py.File(args.filepath, 'r')
keys = list(f.keys())
print('file attrs:', f.attrs.keys())
print('data attrs:', f[keys[0]].attrs.keys())
dkeys = list(f[keys[0]].keys())
for dkey in dkeys:
    print('data key/attr:', f[keys[0]][dkey], f[keys[0]][dkey].attrs.keys())
