import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from statistics import fmean, median, quantiles, pstdev, stdev, pvariance
from neural_model.config import SAMPLE_RATE, BUFFER_SIZE

# load dataset

input_signal_samples = np.load('dataset/input_signal_samples.npy')
output_signal_samples = np.load('dataset/output_signal_samples.npy')


def rms(buffer):
    return np.sqrt(np.mean(buffer ** 2))


def db_rms(rms_value):
    return 10 * np.log10(rms_value)


MAX_POSSIBLE_DB = 24.0
MIN_POSSIBLE_DB = -36.0

samples_rms = np.array([rms(sample) for sample in input_signal_samples], dtype='double')
samples_dbrms = np.nan_to_num(db_rms(samples_rms), neginf=MIN_POSSIBLE_DB, posinf=MAX_POSSIBLE_DB)  # infinite removed

mean_rms = fmean(samples_rms)
median_rms = median(samples_rms)
pstdev_rms = pstdev(samples_rms)
stdev_rms = stdev(samples_rms)
pvariance_rms = pvariance(samples_rms)

print(mean_rms, median_rms, pstdev_rms, stdev_rms, pvariance_rms)

samples_spectres = np.array([np.abs(np.fft.rfft(sample)) for sample in input_signal_samples])
samples_spectral_density = np.array(np.mean(samples_spectres, axis=1))

fig, (ax_rms, ax_db_rms, ax_freq_density) = plt.subplots(nrows=3, constrained_layout=True)

# samples rms, dbrms histograms
ax_rms.hist(samples_rms, color='blue', edgecolor='black', bins=int(256))
ax_db_rms.hist(samples_dbrms, color='green', edgecolor='black', bins=int(256))

test = np.concatenate(input_signal_samples[0: 256])
signal.spectrogram(test, fs=SAMPLE_RATE)

f, t, Sxx = signal.spectrogram(test, SAMPLE_RATE)
ax_freq_density.pcolormesh(t, f, Sxx, shading='gouraud')

# Add labels
ax_rms.set_title('Signal\'s RMS distribution density')
ax_rms.set_xlabel('RMS')
ax_rms.set_ylabel('Amount of samples')

ax_db_rms.set_title('Signal\'s dBRMS distribution density')
ax_db_rms.set_xlabel('dBRMS')
ax_db_rms.set_ylabel('Amount of samples')

ax_freq_density.set_title('Signal\'s frequency mean')
ax_freq_density.set_xlabel('dB')
ax_freq_density.set_ylabel('Frequency')

plt.show()
