from postprocessors.sandbox.near_range import NearRange
import pydarnio


if __name__ == "__main__":

    version = "v1.0"
    file_type = "rawacf"

    processor = NearRange(
        f"/home/remington/repos/borealis_postprocessors/test/{version}/antennas_iq.site",
        "/home/remington/repos/borealis_postprocessors/out.rawacf",
        # f"/home/remington/repos/borealis_postprocessors/test_near_range.bfiq.h5",
        # f"/home/remington/repos/borealis_postprocessors/test_near_range.{file_type}.h5",
        "site",
        "dmap"
    )

    processor.process_file()

    data = pydarnio.read_rawacf("/home/remington/repos/borealis_postprocessors/out.rawacf", mode='strict')

    print(len(data))
