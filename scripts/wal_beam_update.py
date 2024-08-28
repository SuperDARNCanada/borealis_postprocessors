"""
Updates the beam_azms metadata in antennas_iq and rawacf files to reflect the correct beam directions.
"""
import argparse
import glob
import os

import dmap  # `pip install darn-dmap`
import h5py
import numpy as np

from postprocessors.sandbox.update_beam_dirs import UpdateBeamDirs
from postprocessors import borealis_to_borealis_rename


old_spacing = 15.24  # meters
new_spacing = 12.8016  # meters


def hdf5_fixer(path: str):
    structure = get_structure(path)
    if structure == "dmap":
        raise ValueError(f"Unknown structure for file {path}")

    with h5py.File(path, "r+") as f:
        if structure == "array":
            new_beam_dirs = np.rad2deg(
                np.arcsin(
                    old_spacing
                    * np.sin(np.deg2rad(f["beam_azms"][:]))
                    / new_spacing
                )
            )
            f["beam_azms"][:, :] = new_beam_dirs
        else:
            for name in sorted(list(f.keys())):
                rec = f[name]
                new_beam_dirs = np.rad2deg(
                    np.arcsin(
                        old_spacing
                        * np.sin(np.deg2rad(rec["beam_azms"][:]))
                        / new_spacing
                    )
                )
                rec["beam_azms"][:] = new_beam_dirs
    print("  Fixed HDF5 file")


def get_structure(path: str):
    if path.endswith("site"):
        return "site"
    if path.endswith("hdf5"):
        return "array"
    return "dmap"


def dmap_fixer(path: str):
    """Updates the bmazm metadata in a DMAP rawacf file"""
    data = dmap.read_rawacf(path)
    for rec in data:
        rec["bmazm"] = np.rad2deg(
            np.arcsin(old_spacing * np.sin(np.deg2rad(rec["bmazm"])) / new_spacing)
        )
    dmap.write_rawacf(data, path)
    print("  Fixed DMAP file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", help="Directory with files to process")
    parser.add_argument("outdir", help="Directory to store resulting files in (if --reprocess set)")
    parser.add_argument(
        "--pattern", help="Pattern to match filenames", default="*hdf5*"
    )
    parser.add_argument(
        "--reprocess",
        default=False,
        action="store_true",
        help="Flag to reprocess the antennas_iq files to rawacf, rather than just update the metadata",
    )
    parser.add_argument(
        "--for-real",
        action="store_true",
        help="Flag to modify the files (otherwise, just logs what the script *would* do)",
    )
    args = parser.parse_args()

    files = glob.glob(f"{args.indir}/{args.pattern}")
    num_files = len(files)
    failed_files = []

    for i, infile in enumerate(files):
        print(f"{i+1:>5d}/{num_files}  {infile}", end="")

        try:
            if "antennas_iq" in infile:
                if args.for_real:
                    if args.reprocess:
                        processor = UpdateBeamDirs(
                            infile,
                            args.outdir + "/" + borealis_to_borealis_rename(
                                os.path.basename(infile), "rawacf", "site"
                            ),
                            get_structure(infile),
                            "site",
                        )
                        processor.process_file(num_processes=5)
                    else:
                        hdf5_fixer(infile)
                else:
                    if args.reprocess:
                        print("  Reprocessing antennas_iq file")
                    else:
                        print("  Fixing antennas_iq file")

            elif "rawacf" in infile:
                if get_structure(infile) == "dmap":
                    if args.for_real:
                        dmap_fixer(infile)
                    else:
                        print("  Fixing rawacf DMAP file")
                else:
                    if args.for_real:
                        hdf5_fixer(infile)
                    else:
                        print("  Fixing rawacf HDF5 file")

            else:
                raise ValueError(
                    f"Unknown file type for file: {infile}"
                )
        except Exception as e:
            failed_files.append((infile, e))

    if len(failed_files) != 0:
        print(f"\n\nUnable to fix {len(failed_files)} files")
        for fail in failed_files:
            print(f"\t{fail[0]}: {fail[1]}")
