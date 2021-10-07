from random import uniform

import matplotlib.pyplot as plt
from matplotlib import animation

from neural_model.dsp_core.dsp_core import *
from neural_model.config.config import *
from neural_model.config.intervals_config import chords


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


NOISE_AMP_CEIL = 0.2
FRAMES_AMOUNT = 256
AMP_VARIANTS_AMOUNT = 4

for amp_variant in np.linspace(0.2, NORMALIZATION_THRESHOLD, AMP_VARIANTS_AMOUNT):
    for chord in chords.values():
        for wave in [sine_wave, square_wave, sawtooth_wave, pulse_wave, triangular_wave]:
            fig, ax = plt.subplots()

            noise_curve, = ax.plot(
                samples_field,
                chord_additive_signal(chord, wave, samples_field, 10, NOISE_AMP_CEIL),
            )

            plt.ylim([-NORMALIZATION_THRESHOLD, NORMALIZATION_THRESHOLD])


            def animate(i):
                # update the data to plotting
                noise_curve.set_ydata(
                    normalize_filter(
                        chord_additive_signal(chord, wave, samples_field, 10 + 100 * i, NOISE_AMP_CEIL),
                        amp_variant
                    )
                )
                return noise_curve,


            ani = animation.FuncAnimation(
                fig, animate, interval=60, blit=True, save_count=50, frames=FRAMES_AMOUNT
            )

            ax.set(xlabel='buffer samples', ylabel='amplitude', title='Signal buffer')
            ax.grid()
            plt.show()
