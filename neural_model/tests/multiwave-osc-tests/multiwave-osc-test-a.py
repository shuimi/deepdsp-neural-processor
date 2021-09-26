import matplotlib.pyplot as plt
from matplotlib import animation

from neural_model.dsp_core import *

# plotting config
fig, ax = plt.subplots()

multiwave_curve, = ax.plot(samples_field, multiwave_oscillator(samples_field))
time_text = ax.text(0.05, 0.9, 'frame: ', transform=ax.transAxes)
plt.ylim([-NORMALIZATION_THRESHOLD, NORMALIZATION_THRESHOLD])


# modulators
pulse_wave_modulation = np.linspace(-1, 1, 100)

# modulations config
freq_base_modulator_freq = 90


def animate(i):
    # update the data to plotting

    multiwave_curve.set_ydata(
        multiwave_oscillator(
            samples_field,
            freq=NYQUIST_FREQUENCY * triangular_wave(i, freq=200 * sine_wave(i, freq=90)),
            sine_amp=triangular_wave(i, freq=220),
            sine_phase=2*np.pi * sine_wave(i, freq=220),
            sawtooth_amp=sine_wave(i, freq=80),
            sawtooth_phase=2*np.pi * triangular_wave(i, freq=320),
            square_amp=triangular_wave(i, freq=55),
            triangle_amp=sine_wave(i, freq=110),
            pulse_amp=sawtooth_wave(i, freq=440),
            pulse_pwm=pulse_wave_modulation[i % 100],
            normalize=False
        )
    )
    time_text.set_text('frame: ' + str(i))

    return multiwave_curve, time_text


ani = animation.FuncAnimation(
    fig, animate, interval=100, blit=True, save_count=50
)

ax.set(xlabel='buffer samples', ylabel='amplitude', title='Signal buffer')
ax.grid()
plt.show()
