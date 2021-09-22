import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from neural_model.dataset_generation import *
from neural_model.dsp_core import *


model = Sequential()

model.add(Dense(units=BUFFER_SIZE, activation='elu'))
model.add(Dense(units=128, activation='elu'))
model.add(Dense(units=128, activation='elu'))
model.add(Dense(units=128, activation='elu'))
model.add(Dense(units=BUFFER_SIZE, activation='tanh'))

model.compile(
    loss='mean_squared_error',
    optimizer='sgd'
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
    ]
)


# check prediction

test_data = np.array([get_rand_midrange_signal(samples_field, randint(8, 128)) for i in range(BATCH_SIZE)])
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
# TODO: rename plot objects
# TODO: validation split
