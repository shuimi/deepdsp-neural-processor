from random import randint, uniform

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras import backend as K
import matplotlib.pyplot as plt

from neural_model.dataset_generation import chord_additive_signal, approximation_target_filter
from neural_model.intervals_config import chords
from neural_model.dsp_core import *


def activation(x):
    return NORMALIZATION_THRESHOLD * 2 * K.sigmoid(0.25 * x) - 1


# load dataset

input_signal_samples = np.load('dataset/input_signal_samples.npy')
output_signal_samples = np.load('dataset/output_signal_samples.npy')

# setup model
model = Sequential([
    Dense(units=BUFFER_SIZE, activation=activation),
    Dense(units=128, activation=activation),
    Dropout(rate=0.02),
    Dense(units=128, activation=activation),
    Dropout(rate=0.02),
    Dense(units=BUFFER_SIZE, activation=activation)
])


model.compile(
    loss='mean_squared_error',
    optimizer='rmsprop',
    metrics=['accuracy']
)

# train model

model.fit(
    input_signal_samples,
    output_signal_samples,
    epochs=EPOCHS_AMOUNT,
    batch_size=BATCH_SIZE,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA
        )
    ],
    validation_split=0.2
)


print(model.summary())


# generate test data

test_data = []

for _ in range(BATCH_SIZE):

    chord = list(chords.values())[randint(0, len(chords.values()) - 1)]
    wave = [sine_wave, square_wave, sawtooth_wave, pulse_wave, triangular_wave][randint(0, 4)]
    noise_max_amp = uniform(0, 0.5)
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
    return np.max(np.absolute(init_buffer) - np.absolute(processed_buffer))


model.save('model.h5')


tf.keras.utils.plot_model(
    model,
    to_file="model_graph.png",
    show_shapes=False,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
    layer_range=None,
)


for i in range(BATCH_SIZE):
    fig, ax = plt.subplots()

    ax.plot(samples_field, processed[i])
    ax.plot(samples_field, prediction[i])

    ax.set(xlabel='time (s)', ylabel='dB', title='Signal')
    ax.text(0.05, 0.9, 'max error: ' + str(max_error(processed[i], prediction[i])), transform=ax.transAxes)

    ax.grid()
    plt.show()


# TODO: add signal audio preview
