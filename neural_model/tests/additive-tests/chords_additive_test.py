from random import uniform

import matplotlib.pyplot as plt
from matplotlib import animation

from neural_model.dsp_core.dsp_core import *
from neural_model.config.config import *
from neural_model.config.intervals_config import chords

NOISE_AMP_CEIL = 0.2


def chord_additive_signal(chord, wave, signal_buffer, freq, noise_max_amp):
    signal = np.zeros(len(signal_buffer))

    for interval in chord:
        signal = signal + wave(
            signal_buffer,
            freq=freq * interval,
            amp=uniform(-NORMALIZATION_THRESHOLD, NORMALIZATION_THRESHOLD)
        )

    signal = normalize_filter(signal, NORMALIZATION_THRESHOLD)
    return normalize_filter(signal + white_noise(signal_buffer, uniform(0, noise_max_amp)), NORMALIZATION_THRESHOLD)


fig, ax = plt.subplots()

time_text = ax.text(0.05, 0.9, 'frame: ', transform=ax.transAxes)

noise_curve, = ax.plot(
    samples_field,
    chord_additive_signal(chords['minor_triad'], sawtooth_wave, samples_field, 440, NOISE_AMP_CEIL)
)

plt.ylim([-NORMALIZATION_THRESHOLD, NORMALIZATION_THRESHOLD])


def animate(i):
    # update the data to plotting
    noise_curve.set_ydata(
        chord_additive_signal(chords['minor_triad'], sine_wave, samples_field, 10 + 20 * i, NOISE_AMP_CEIL)
    )
    time_text.set_text('frame: ' + str(i))

    return noise_curve, time_text


ani = animation.FuncAnimation(
    fig, animate, interval=60, blit=True, save_count=50
)

ax.set(xlabel='buffer samples', ylabel='amplitude', title='Signal buffer')
ax.grid()
plt.show()
