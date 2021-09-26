import matplotlib.pyplot as plt
from matplotlib import animation

from neural_model.dsp_core import *

# test config
BASE_FREQUENCY_HZ = 440.0
FREQUENCY_STEP_HZ = 5.0

SINE_BASE_AMP = 1.0
SAWTOOTH_BASE_AMP = 1.5
TRIANGULAR_BASE_AMP = 2.0
SQUARE_BASE_AMP = 2.5
PULSE_BASE_AMP = 3.0

# waves animated plotting (with linear freq modulation)

fig, ax = plt.subplots()

sine_curve, = ax.plot(
    samples_field,
    sine_wave(samples_field, freq=BASE_FREQUENCY_HZ, amp=SINE_BASE_AMP)
)
sawtooth_curve, = ax.plot(
    samples_field,
    sawtooth_wave(samples_field, freq=BASE_FREQUENCY_HZ, amp=SAWTOOTH_BASE_AMP, phase=4 * np.pi)
)
triangular_curve, = ax.plot(
    samples_field,
    triangular_wave(samples_field, freq=BASE_FREQUENCY_HZ, amp=TRIANGULAR_BASE_AMP)
)
square_curve, = ax.plot(
    samples_field,
    square_wave(samples_field, freq=BASE_FREQUENCY_HZ, amp=SQUARE_BASE_AMP)
)
pulse_curve, = ax.plot(
    samples_field,
    pulse_wave(samples_field, freq=BASE_FREQUENCY_HZ, amp=PULSE_BASE_AMP, pwm=0.0)
)

plt.ylim([-NORMALIZATION_THRESHOLD, NORMALIZATION_THRESHOLD])
pulse_wave_modulation = np.linspace(-1, 1, 100)


def animate(i):
    # update the data to plotting
    sine_curve.set_ydata(
        sine_wave(samples_field, freq=BASE_FREQUENCY_HZ + FREQUENCY_STEP_HZ * i, amp=SINE_BASE_AMP)
    )
    sawtooth_curve.set_ydata(
        sawtooth_wave(
            samples_field,
            freq=BASE_FREQUENCY_HZ + FREQUENCY_STEP_HZ * i, amp=SAWTOOTH_BASE_AMP, phase=4 * np.pi
        )
    )
    triangular_curve.set_ydata(
        triangular_wave(samples_field, freq=BASE_FREQUENCY_HZ + FREQUENCY_STEP_HZ * i, amp=TRIANGULAR_BASE_AMP)
    )
    square_curve.set_ydata(
        square_wave(samples_field, freq=BASE_FREQUENCY_HZ + FREQUENCY_STEP_HZ * i, amp=SQUARE_BASE_AMP)
    )
    pulse_curve.set_ydata(
        pulse_wave(
            samples_field, freq=BASE_FREQUENCY_HZ + FREQUENCY_STEP_HZ * i, amp=PULSE_BASE_AMP,
            pwm=pulse_wave_modulation[i % 100]
        )
    )
    return sine_curve, sawtooth_curve, triangular_curve, square_curve, pulse_curve


ani = animation.FuncAnimation(
    fig, animate, interval=20, blit=True, save_count=50
)

ax.set(xlabel='buffer samples', ylabel='amplitude', title='Signal buffer')
ax.grid()
plt.show()
