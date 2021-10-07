import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras import backend as K
from neural_model.dsp_core.dsp_core import *

EARLY_STOPPING_PATIENCE = 4
EARLY_STOPPING_MIN_DELTA = 0.001


def activation(x):
    return NORMALIZATION_THRESHOLD * 2 * K.sigmoid(0.25 * x) - 1


# load dataset

input_signal_samples = np.load('../dataset/input_signal_samples.npy')
output_signal_samples = np.load('../dataset/output_signal_samples.npy')

# setup model
model = Sequential([
    Dense(units=BUFFER_SIZE, activation=activation),
    Dense(units=512, activation=activation),
    Dropout(rate=0.3),
    Dense(units=256, activation=activation),
    Dropout(rate=0.2),
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
model.save('model.h5')

