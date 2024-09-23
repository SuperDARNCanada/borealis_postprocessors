"""
Reprocesses antennas_iq -> rawacf assuming a uniform 3.91 degree beam separation and updated antenna spacing.
"""
import argparse
import glob
import os

from postprocessors.sandbox.update_beam_dirs import UpdateBeamDirs
from postprocessors import borealis_to_borealis_rename


def get_structure(path: str):
    if path.endswith("site"):
        return "site"
    if path.endswith("hdf5"):
        return "array"
    return "dmap"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", help="Directory with files to process")
    parser.add_argument("outdir", help="Directory to store resulting files in")
    parser.add_argument(
        "--pattern", help="Pattern to match filenames", default="*antennas_iq.hdf5*"
    )
    parser.add_argument(
        "--for-real",
        action="store_true",
        help="Flag to modify the files (otherwise, just logs what the script *would* do)",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        help="Number of processes to use when reprocessing antennas_iq -> rawacf. Default 1",
        default=1
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
                    processor = UpdateBeamDirs(
                        infile,
                        args.outdir + "/" + borealis_to_borealis_rename(
                            os.path.basename(infile), "rawacf", "site"
                        ),
                        get_structure(infile),
                        "site",
                    )
                    processor.process_file(num_processes=args.num_processes)
                else:
                    print("  Reprocessing antennas_iq file")

            else:
                raise ValueError(
                    f"Unsupported file type for file: {infile}"
                )
        except Exception as e:
            failed_files.append((infile, e))

    if len(failed_files) != 0:
        print(f"\n\nUnable to fix {len(failed_files)} files")
        for fail in failed_files:
            print(f"\t{fail[0]}: {fail[1]}")
