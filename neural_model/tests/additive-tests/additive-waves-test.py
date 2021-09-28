from random import randint, uniform

import matplotlib.pyplot as plt
from matplotlib import animation

from neural_model.dsp_core import *


def rand_additive_signal(signal_buffer, harmonics_amount, low_freq_bound, hi_freq_bound, noise_max_amp, wave):
    signal = np.zeros(len(signal_buffer))

    for harmonic in range(harmonics_amount):
        signal = signal + wave(
            signal_buffer,
            freq=uniform(low_freq_bound, hi_freq_bound),
            phase=2 * np.pi * random(),
            amp=uniform(-NORMALIZATION_THRESHOLD, NORMALIZATION_THRESHOLD)
        )

    signal = normalize_filter(signal, NORMALIZATION_THRESHOLD)
    return normalize_filter(signal + white_noise(signal_buffer, uniform(0, noise_max_amp)), NORMALIZATION_THRESHOLD)


filter_decay_modulation = np.linspace(0, 1, 100)

fig, ax = plt.subplots()

noise_curve, = ax.plot(
    samples_field,
    lowpass(
        rand_additive_signal(samples_field, randint(1, 4), 10, 400, 0.1, square_wave),
        filter_decay_modulation[0]
    )
)

plt.ylim([-NORMALIZATION_THRESHOLD, NORMALIZATION_THRESHOLD])


def animate(i):
    # update the data to plotting
    noise_curve.set_ydata(
        lowpass(
            rand_additive_signal(samples_field, randint(1, 4), 10, 400, 0.1, square_wave),
            filter_decay_modulation[i % 100]
        )
    )
    return noise_curve,


ani = animation.FuncAnimation(
    fig, animate, interval=60, blit=True, save_count=50
)

ax.set(xlabel='buffer samples', ylabel='amplitude', title='Signal buffer')
ax.grid()
plt.show()
