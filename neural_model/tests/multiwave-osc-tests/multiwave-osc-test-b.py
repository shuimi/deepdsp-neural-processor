import matplotlib.pyplot as plt
from matplotlib import animation

from neural_model.dsp_core import *


# plotting config
fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 6))

ax1.set_ylim([-NORMALIZATION_THRESHOLD, NORMALIZATION_THRESHOLD])

multiwave_curve, = ax1.plot(samples_field, multiwave_oscillator(samples_field))
ax2.magnitude_spectrum([0], Fs=SAMPLE_RATE, scale='dB')


# modulators
pulse_wave_modulation = np.linspace(-1, 1, 100)
filter_decay_modulation = np.linspace(0, 1, 100)


def animate(i):
    # update the data to plotting

    signal = normalize_filter(
        lowpass(
            multiwave_oscillator(
                samples_field,
                freq=NYQUIST_FREQUENCY * sawtooth_wave(i, freq=200 * sine_wave(i, freq=90)),
                sine_amp=triangular_wave(i, freq=220),
                sine_phase=2 * np.pi * square_wave(i, freq=220),
                sawtooth_amp=sine_wave(i, freq=80),
                sawtooth_phase=2 * np.pi * triangular_wave(i, freq=320),
                square_amp=triangular_wave(i, freq=55),
                triangle_amp=sawtooth_wave(i, freq=1110),
                pulse_amp=sawtooth_wave(i, freq=440),
                pulse_pwm=pulse_wave_modulation[i % 100],
                normalize=False
            )
            + white_noise(samples_field, 0.5),
            filter_decay_modulation[i % 100]
        ),
        NORMALIZATION_THRESHOLD
    )

    multiwave_curve.set_ydata(signal)
    _, _, line = ax2.magnitude_spectrum(signal, Fs=SAMPLE_RATE, scale='dB', color='C1')

    return multiwave_curve, line,


ani = animation.FuncAnimation(
    fig, animate, interval=70, blit=True, save_count=50
)

fig.tight_layout()
plt.show()
