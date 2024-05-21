import numpy as np
import numpy.fft as fft
import glob
import matplotlib.pyplot as plt
import h5py
import argparse


def plot_array_data(infile):
    record = h5py.File(infile, 'r')
    dataset = record["data"]

    fs = 5e6 / 15  # Hz

    pulses = record["pulses"]
    tau = round(record.attrs["tau_spacing"] * 1e-6 * fs)  # puts into units of samples
    pulses_in_samples = [int(round(p * tau)) for p in pulses]
    pulse_dur = round(0.0006 * fs)
    mask = np.zeros(dataset.shape[-1], dtype=bool)
    for pulse in pulses_in_samples:
        start_mask = int(round(pulse - pulse_dur / 2))
        end_mask = int(round(pulse + pulse_dur / 2))
        if start_mask < 0:
            start_mask = 0
        if end_mask >= len(mask):
            end_mask = len(mask) - 1
        mask[start_mask: end_mask] = 1

    num_sequences = dataset.shape[0]
    num_samps = dataset.shape[-1]
    fft_data = []
    for sqn in range(num_sequences):
        data = dataset[sqn, ...]
        data[..., mask] = 0.0 + 0.0j
        fft_data.append(np.sum(np.abs(fft.fftshift(fft.fft(data), axes=-1)), axis=0))
    fft_data = np.array(fft_data)[:, 0, :]

    freqs = fft.fftshift(fft.fftfreq(num_samps, d=1 / fs))
    df = freqs[1] - freqs[0]
    output_freq_resolution = 1000  # Hz
    sample_resolution = int(round(output_freq_resolution / df))
    kernel = np.ones((sample_resolution,)) / sample_resolution

    filtered_fft = np.array([np.convolve(fft_data[i, :], kernel, mode='same')
                            for i in range(num_sequences)])
    np.save(infile.replace(".antennas_iq.hdf5", ".filtered_fft"), filtered_fft)

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    img = ax.imshow(
        10 * np.log10(fft_data),
        aspect='auto', origin='lower',
        cmap=plt.get_cmap('plasma'),
        vmax=20, vmin=-10,
        extent=[freqs[0] - df / 2, freqs[-1] + df / 2, -0.5, fft_data.shape[0] + 0.5],
    )
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Sequence")
    fig.colorbar(img, ax=ax, label="Power [dB]")

    plt.savefig(infile.replace(".antennas_iq.hdf5", ".jpg"), bbox_inches="tight")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Directory with files to process")
    parser.add_argument("pattern", help="Pattern for globbing files")
    args = parser.parse_args()

    files = glob.glob(f"{args.dir}/{args.pattern}")
    for f in files:
        if f.endswith(".site"):
            continue
        # print(f"plot_array_data({f})")
        plot_array_data(f)
