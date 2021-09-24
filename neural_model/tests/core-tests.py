import matplotlib.pyplot as plt
from matplotlib import animation

from neural_model.config import NORMALIZATION_THRESHOLD
from neural_model.dsp_core import *

# sawtooth plot

fig, ax = plt.subplots()
line, = ax.plot(samples_field, sine_wave(samples_field, freq=20.0))
plt.ylim([-NORMALIZATION_THRESHOLD, NORMALIZATION_THRESHOLD])


def animate(i):
    line.set_ydata(sine_wave(samples_field, freq=20.0 + 1 * i))  # update the data.
    return line,


ani = animation.FuncAnimation(
    fig, animate, interval=20, blit=True, save_count=50
)

ax.set(xlabel='buffer samples', ylabel='amplitude', title='Signal buffer')
ax.grid()
plt.show()
