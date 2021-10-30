import matplotlib.pyplot as plt
from random import randint, uniform

from keras.layers import Activation

from neural_model.config.intervals_config import chords
from neural_model.dataset.generation_tools import *
from neural_model.dsp_core.dsp_core import *

from tensorflow import keras
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

import tensorflow as tf


def activation(x):
    return NORMALIZATION_THRESHOLD * 2 * K.sigmoid(0.25 * x) - 1


def pre_emphasis_filter(x, coeff=0.85):
    return tf.concat([x[:, 0:1, :], x[:, 1:, :] - coeff * x[:, :-1, :]], axis=1)


def dc_loss(target_y, predicted_y):
    return tf.reduce_mean(
        tf.square(tf.reduce_mean(target_y) - tf.reduce_mean(predicted_y))
    ) / tf.reduce_mean(tf.square(target_y))


def esr_loss(target_y, predicted_y, emphasis_func=lambda x: x):
    target_yp = emphasis_func(target_y)
    pred_yp = emphasis_func(predicted_y)
    return tf.reduce_sum(tf.square(target_yp - pred_yp)) / tf.reduce_sum(tf.square(target_yp))


get_custom_objects().update({'activation': Activation(activation)})
get_custom_objects().update({'esr_loss': esr_loss})


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


# test_data = np.expand_dims(test_data, axis=2)
# processed = np.expand_dims(processed, axis=2)


# check prediction

prediction = model.predict_on_batch(test_data)


def max_error(init_buffer, processed_buffer):
    return np.max(np.absolute(init_buffer - processed_buffer))


for i in range(BATCH_SIZE):
    fig, (ax_target, ax_delta, ax_source) = plt.subplots(nrows=3, constrained_layout=True)

    processed_line, prediction_line_1, = ax_target.plot(samples_field, processed[i], prediction[i])
    delta_line, = ax_delta.plot(samples_field, processed[i] - prediction[i])
    test_line, prediction_line_2, = ax_source.plot(samples_field, test_data[i], prediction[i])

    ax_target.text(
        0.05, 0.9, 'max error: ' + str(max_error(processed[i], prediction[i])), transform=ax_target.transAxes
    )

    ax_target.set(xlabel='Samples', ylabel='Amplitude', title='Signal: target, prediction')
    ax_delta.set(xlabel='Samples', ylabel='Amplitude', title='Signal delta: target, prediction')
    ax_source.set(xlabel='Samples', ylabel='Amplitude', title='Signal: source, prediction')

    ax_target.legend([processed_line, prediction_line_1], ['Target processed signal', 'Model prediction'])
    ax_delta.legend([delta_line], ['Delta: processed - predicted'])
    ax_source.legend([test_line, prediction_line_2], ['Unprocessed signal', 'Model prediction'])

    ax_target.grid()
    ax_delta.grid()
    ax_source.grid()
    plt.show()
