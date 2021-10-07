import matplotlib.pyplot as plt
from random import randint, uniform

from keras.layers import Activation

from neural_model.dataset_generation import chord_additive_signal, approximation_target_filter
from neural_model.intervals_config import chords
from neural_model.dsp_core import *

from tensorflow import keras
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


def activation(x):
    return NORMALIZATION_THRESHOLD * 2 * K.sigmoid(0.25 * x) - 1


get_custom_objects().update({'activation': Activation(activation)})

model = keras.models.load_model('model.h5')

# generate test data

test_data = []

for _ in range(BATCH_SIZE):

    chord = list(chords.values())[randint(0, len(chords.values()) - 1)]
    wave = [sine_wave, square_wave, sawtooth_wave, pulse_wave, triangular_wave][randint(0, 4)]
    noise_max_amp = uniform(0, 0.3)
    base_freq = uniform(10, 12000)
    norm_amp = uniform(0.1, 4.0)

    test_data += [
        normalize_filter(
            chord_additive_signal(chord, wave, samples_field, base_freq, noise_max_amp),
            norm_amp
        )
    ]

test_data = np.array(test_data)

processed = np.array([approximation_target_filter(signal) for signal in test_data])


# check prediction

prediction = model.predict_on_batch(test_data)


def max_error(init_buffer, processed_buffer):
    return np.max(np.absolute(init_buffer - processed_buffer))


for i in range(BATCH_SIZE):
    fig, (ax_target, ax_source) = plt.subplots(nrows=2, constrained_layout=True)

    processed_line, prediction_line_1, = ax_target.plot(samples_field, processed[i], prediction[i])
    test_line, prediction_line_2, = ax_source.plot(samples_field, test_data[i], prediction[i])

    ax_target.text(
        0.05, 0.9, 'max error: ' + str(max_error(processed[i], prediction[i])), transform=ax_target.transAxes
    )

    ax_target.set(xlabel='Samples', ylabel='Amplitude', title='Signal: target, prediction')
    ax_source.set(xlabel='Samples', ylabel='Amplitude', title='Signal: source, prediction')

    ax_target.legend([processed_line, prediction_line_1], ['Target processed signal', 'Model prediction'])
    ax_source.legend([test_line, prediction_line_2], ['Unprocessed signal', 'Model prediction'])

    ax_target.grid()
    ax_source.grid()
    plt.show()
