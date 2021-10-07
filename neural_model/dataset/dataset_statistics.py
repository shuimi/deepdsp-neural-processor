import matplotlib.pyplot as plt
from statistics import fmean, median, stdev, pvariance
from neural_model.dsp_core.dsp_core import *

# load dataset

input_signal_samples = np.load('input_signal_samples.npy')
output_signal_samples = np.load('output_signal_samples.npy')


MAX_POSSIBLE_DB = 24.0
MIN_POSSIBLE_DB = -36.0

samples_peaks = np.array([np.max(np.abs(sample)) for sample in input_signal_samples])
samples_rms = np.array([rms(sample) for sample in input_signal_samples], dtype='double')
samples_dbrms = np.nan_to_num(db_rms(samples_rms), neginf=MIN_POSSIBLE_DB, posinf=MAX_POSSIBLE_DB)  # infinite removed

mean_rms = fmean(samples_rms)
median_rms = median(samples_rms)
stdev_rms = stdev(samples_rms)
pvariance_rms = pvariance(samples_rms)

print(mean_rms, median_rms, stdev_rms, pvariance_rms)

fig, (ax_rms, ax_db_rms, ax_peaks) = plt.subplots(nrows=3, constrained_layout=True)

# samples rms, dbrms histograms
ax_rms.hist(samples_rms, color='blue', edgecolor='black', bins=int(256))
ax_db_rms.hist(samples_dbrms, color='green', edgecolor='black', bins=int(256))
ax_peaks.hist(samples_peaks, color='yellow', edgecolor='black', bins=int(256))

# Add labels
ax_rms.set_title('Signal\'s RMS distribution density')
ax_rms.set_xlabel('RMS')
ax_rms.set_ylabel('Amount of samples')

ax_db_rms.set_title('Signal\'s dBRMS distribution density')
ax_db_rms.set_xlabel('dBRMS')
ax_db_rms.set_ylabel('Amount of samples')

ax_peaks.set_title('Signal\'s Peaks distribution density')
ax_peaks.set_xlabel('Peak')
ax_peaks.set_ylabel('Amount of samples')

plt.show()
