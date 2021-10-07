import matplotlib.pyplot as plt
from matplotlib import animation

from neural_model.dsp_core.dsp_core import *


def windowed_sinc_filter(cutoff_freq=0.05, filter_length=15):
    # compute sinc filter
    sinc_function = np.sinc(2 * cutoff_freq * (np.arange(filter_length) - (filter_length - 1) / 2))

    # apply window
    sinc_function *= np.hamming(filter_length)

    # normalize to get unity gain
    sinc_function /= np.sum(sinc_function)

    def lowpass_filter(signal_buffer):
        return np.convolve(signal_buffer, sinc_function)[0:BUFFER_SIZE]

    return lowpass_filter


# config
WHITE_NOISE_MAX_AMP = 2.0

low_pass_filter = windowed_sinc_filter(cutoff_freq=0.01, filter_length=31)
bias_wave_modulation = np.linspace(-2, 2, 100)

# waves animated plotting (with linear freq modulation)

fig, ax = plt.subplots()

noise_curve, = ax.plot(
    samples_field,
    low_pass_filter(white_noise(samples_field, max_amp=WHITE_NOISE_MAX_AMP))
)

plt.ylim([-NORMALIZATION_THRESHOLD, NORMALIZATION_THRESHOLD])


def animate(i):
    # update the data to plotting
    noise_curve.set_ydata(
        low_pass_filter(white_noise(samples_field, max_amp=WHITE_NOISE_MAX_AMP)) +
        bias_wave(bias_wave_modulation[i % 100])
    )
    return noise_curve,


ani = animation.FuncAnimation(
    fig, animate, interval=16, blit=True, save_count=50
)

ax.set(xlabel='buffer samples', ylabel='amplitude', title='Signal buffer')
ax.grid()
plt.show()
