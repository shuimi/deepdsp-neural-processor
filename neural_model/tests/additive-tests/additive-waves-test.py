from random import randint

import matplotlib.pyplot as plt
from matplotlib import animation

from neural_model.dataset_generation import rand_additive_signal, filter_decay_modulation
from neural_model.dsp_core import *


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
