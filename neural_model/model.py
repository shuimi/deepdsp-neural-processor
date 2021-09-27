import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from neural_model.dataset_generation import *
from neural_model.dsp_core import *


model = Sequential()

model.add(Dense(units=BUFFER_SIZE, activation='elu'))
model.add(Dense(units=128, activation='elu'))
model.add(Dropout(rate=0.02))
model.add(Dense(units=128, activation='elu'))
model.add(Dense(units=BUFFER_SIZE, activation='tanh'))

model.compile(
    loss='mean_squared_error',
    optimizer='rmsprop'
)
# 65536
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


# check prediction

test_data = np.array([rand_additive_signal(samples_field, 12, 50, 8000, 1.0, sine_wave) for i in range(BATCH_SIZE)])
clipped = np.array([signal_clipping_filter(signal, sample_hard_clip) for signal in test_data])
prediction = model.predict_on_batch(test_data)

for i in range(BATCH_SIZE):
    fig, ax = plt.subplots()

    ax.plot(samples_field, clipped[i])
    ax.plot(samples_field, prediction[i])

    ax.set(xlabel='time (s)', ylabel='dB', title='Signal')
    ax.grid()

    # fig.savefig("signal.png")
    plt.show()

# TODO: add dropouts
# TODO: add metrics
# TODO: rename plot objects
# TODO: validation split
# TODO: export model graph
