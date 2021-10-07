from random import uniform

import matplotlib.pyplot as plt
from matplotlib import animation

from neural_model.dsp_core.dsp_core import *


def rand_additive_sine_wave_signal(signal_buffer, harmonics_amount, low_freq_bound, hi_freq_bound, noise_max_amp):
    signal = np.zeros(len(signal_buffer))

    for harmonic in range(harmonics_amount):
        signal = signal + sine_wave(
            signal_buffer,
            freq=uniform(low_freq_bound, hi_freq_bound),
            phase=2 * np.pi * random(),
            amp=uniform(-NORMALIZATION_THRESHOLD, NORMALIZATION_THRESHOLD)
        )

    signal = normalize_filter(signal, NORMALIZATION_THRESHOLD)
    return normalize_filter(signal + white_noise(uniform(signal_buffer, noise_max_amp)), NORMALIZATION_THRESHOLD)


fig, ax = plt.subplots()

noise_curve, = ax.plot(
    samples_field,
    rand_additive_sine_wave_signal(samples_field, 12, 50, 8000, 1.0)
)

plt.ylim([-NORMALIZATION_THRESHOLD, NORMALIZATION_THRESHOLD])


def animate(i):
    # update the data to plotting
    noise_curve.set_ydata(
        rand_additive_sine_wave_signal(samples_field, 12, 50, 8000, 1.0)
    )
    return noise_curve,


ani = animation.FuncAnimation(
    fig, animate, interval=60, blit=True, save_count=50
)

ax.set(xlabel='buffer samples', ylabel='amplitude', title='Signal buffer')
ax.grid()
plt.show()
