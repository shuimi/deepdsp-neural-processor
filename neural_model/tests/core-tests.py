import matplotlib.pyplot as plt
from matplotlib import animation

from neural_model.dsp_core import *

# config
BASE_FREQUENCY_HZ = 440.0
FREQUENCY_STEP_HZ = 1.0

# sawtooth plot

fig, ax = plt.subplots()

sine_curve, = ax.plot(samples_field, sine_wave(samples_field, freq=BASE_FREQUENCY_HZ))
sawtooth_curve, = ax.plot(samples_field, sawtooth_wave(samples_field, freq=BASE_FREQUENCY_HZ))
triangular_curve, = ax.plot(samples_field, triangular_wave(samples_field, freq=BASE_FREQUENCY_HZ))
square_curve, = ax.plot(samples_field, square_wave(samples_field, freq=BASE_FREQUENCY_HZ))

plt.ylim([-NORMALIZATION_THRESHOLD, NORMALIZATION_THRESHOLD])


def animate(i):
    # update the data to plotting
    sine_curve.set_ydata(
        sine_wave(samples_field, freq=BASE_FREQUENCY_HZ + FREQUENCY_STEP_HZ * i)
    )
    sawtooth_curve.set_ydata(
        sawtooth_wave(samples_field, freq=BASE_FREQUENCY_HZ + FREQUENCY_STEP_HZ * i)
    )
    triangular_curve.set_ydata(
        triangular_wave(samples_field, freq=BASE_FREQUENCY_HZ + FREQUENCY_STEP_HZ * i)
    )
    square_curve.set_ydata(
        square_wave(samples_field, freq=BASE_FREQUENCY_HZ + FREQUENCY_STEP_HZ * i)
    )
    return sine_curve, sawtooth_curve, triangular_curve, square_curve


ani = animation.FuncAnimation(
    fig, animate, interval=20, blit=True, save_count=50
)

ax.set(xlabel='buffer samples', ylabel='amplitude', title='Signal buffer')
ax.grid()
plt.show()
