import matplotlib.pyplot as plt
from scipy.signal import convolve

from neural_model.dsp_core.dsp_core import *


signal = square_wave(samples_field, freq=1400.0) + sawtooth_wave(samples_field, freq=440)
filtered_signal = butter_bandpass_filter(signal, cutoff_low=800)
convolved_signals = normalize_filter(
    convolve(square_wave(samples_field, freq=668), triangular_wave(samples_field, freq=1900))[0:128],
    NORMALIZATION_THRESHOLD
)

plt.plot(samples_field, signal, 'b-', label='data')
plt.plot(samples_field, filtered_signal, 'g-', linewidth=2, label='filtered data')
plt.plot(samples_field, convolved_signals, 'm-', linewidth=2, label='filtered data')
plt.grid()
plt.show()
