import matplotlib.pyplot as plt
import simpleaudio as sa
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from scipy import signal

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation, LSTM, Reshape, Conv1D, MaxPooling1D, Flatten

from keras import backend as keras_backend
from keras.utils.generic_utils import get_custom_objects

from neural_model.config.intervals_config import chords
from neural_model.dataset.generation_tools import *
from neural_model.dsp_core.dsp_core import *


# %%
def activation(x):
    return NORMALIZATION_THRESHOLD * 2 * keras_backend.sigmoid(0.25 * x) - 1


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


# %%
get_custom_objects().update({
    'activation': Activation(activation),
    'esr_loss': esr_loss
})


# %%
class Model:

    def __init__(self, input_data, output_data, optimizer, loss):
        self.input_samples = input_data
        self.output_samples = output_data
        self.optimizer = optimizer
        self.loss = loss

        self.model = Sequential()

    def compile(self):
        # self.model.add(Reshape(
        #     target_shape=(self.input_samples.shape[1], 1)
        # ))
        # self.model.add(
        #     Conv1D(
        #         filters=32, kernel_size=5, strides=1, kernel_initializer='uniform',
        #         # input_shape=input_signal_samples.shape[1:],
        #         activation=activation
        #     )
        # )
        # self.model.add(
        #     Conv1D(
        #         filters=64, kernel_size=5, strides=1, kernel_initializer='uniform',
        #         # input_shape=input_signal_samples.shape[1:],
        #         activation=activation
        #     )
        # )
        # self.model.add(MaxPooling1D(data_format='channels_last'))
        # self.model.add(Flatten())

        # self.model.add(Dense(units=256, activation=activation))
        # self.model.add(Dropout(rate=0.2))

        self.model.add(Reshape(
            target_shape=(self.input_samples.shape[1], 1)
        ))
        self.model.add(LSTM(units=90, activation=activation, dropout=0.1))
        self.model.add(Dense(units=BUFFER_SIZE))

        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=['mse', 'accuracy']
        )

    def build(self):
        self.model.build(input_shape=self.input_samples.shape)

    def summary(self):
        return self.model.summary()

    def train(self, epochs_amount, batch_size, callbacks):
        self.model.fit(
            self.input_samples,
            self.output_samples,
            epochs=epochs_amount,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_split=0.2
        )

    @staticmethod
    def single_test(model, test_data):

        processed = approximation_target_filter(test_data)

        # check prediction
        prediction = model.predict(np.reshape(test_data, newshape=(1, BUFFER_SIZE)), batch_size=1)[0]

        def max_error(init_buffer, processed_buffer):
            return np.max(np.absolute(init_buffer - processed_buffer))

        fig, (ax_target, ax_delta, ax_source) = plt.subplots(nrows=3, constrained_layout=True)

        processed_line, prediction_line_1, = ax_target.plot(samples_field, processed, prediction)
        delta_line, = ax_delta.plot(samples_field, processed - prediction)
        test_line, prediction_line_2, = ax_source.plot(samples_field, test_data, prediction)

        ax_target.text(
            0.02, 0.08, 'max error: ' + str(max_error(processed, prediction)), transform=ax_target.transAxes
        )
        ax_delta.text(
            0.02, 0.08, 'delta rms: ' + str(rms(processed - prediction)), transform=ax_delta.transAxes
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

    @staticmethod
    def audio_test(model):
        print(model.summary())

        seconds = 2
        time = np.arange(0, SAMPLE_RATE * seconds, 1)

        raw_audio = chord_additive_signal(
            chords['perfect_5th'], wave=square_wave, signal_buffer=time, noise_max_amp=0.2, freq=110*1.5
        )
        target_audio = approximation_target_filter(raw_audio)

        data_to_model = np.split(
            raw_audio[0: BUFFER_SIZE * ((seconds * SAMPLE_RATE) // BUFFER_SIZE): 1],
            (seconds * SAMPLE_RATE) // BUFFER_SIZE
        )
        data_to_model = np.expand_dims(data_to_model, axis=1)

        predicted_audio = model.predict(data_to_model, batch_size=len(data_to_model))
        predicted_audio = predicted_audio.flatten()

        f, t, Sxx = signal.spectrogram(target_audio, SAMPLE_RATE)
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

        play = sa.play_buffer(
            to16bit(raw_audio),
            num_channels=1,
            bytes_per_sample=2,
            sample_rate=SAMPLE_RATE
        )
        play.wait_done()

        play = sa.play_buffer(
            to16bit(target_audio),
            num_channels=1,
            bytes_per_sample=2,
            sample_rate=SAMPLE_RATE
        )
        play.wait_done()

        play = sa.play_buffer(
            to16bit(predicted_audio),
            num_channels=1,
            bytes_per_sample=2,
            sample_rate=SAMPLE_RATE
        )
        play.wait_done()

    def save_model(self, model_name):
        self.model.save(model_name)


import random
import string
from datetime import datetime


# %%
def random_code(length):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))


inherits_from = 'model_3GZZ6'
model_id = random_code(5)
model_path = f'models/model_{model_id}/model_{model_id}'

model = tf.keras.models.load_model('models/model_3GZZ6/model_3GZZ6.h5')

# model = Model(
#     input_data=np.load('../dataset/input_signal_samples.npy'),
#     output_data=np.load('../dataset/output_signal_samples.npy'),
#     optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4, epsilon=1e-3),
#     loss=esr_loss
# )

model.compile(
    metrics=['mse', 'accuracy'],
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4, epsilon=1e-3),
    loss=esr_loss
)
model.build()

print(
    f'MODEL_ID:{model_id}\n',
    model.summary()
)


log_dir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")

early_stopping_callback = EarlyStopping(monitor='loss', patience=4, min_delta=0.0005)
reduce_lr_callback = ReduceLROnPlateau(monitor='loss', patience=2, factor=0.2, min_lr=0.001)
tensorboard_callback = TensorBoard(log_dir=log_dir)

model.fit(
    epochs=EPOCHS_AMOUNT,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping_callback, reduce_lr_callback, tensorboard_callback],
    x=np.load('../dataset/input_signal_samples.npy'),
    y=np.load('../dataset/output_signal_samples.npy')
)

# model.train(
#     epochs_amount=EPOCHS_AMOUNT,
#     batch_size=BATCH_SIZE,
#     callbacks=[early_stopping_callback, reduce_lr_callback, tensorboard_callback]
# )

model.save_model(model_path + '_inherits_from_' + inherits_from + '.h5')
with open(model_path + '.txt', 'w') as file:
    file.write(model.model.to_json())


# Model.single_test(
#     model=tf.keras.models.load_model('models/model_3GZZ6/model_3GZZ6.h5'),
#     test_data=chord_additive_signal(
#                 chords['perfect_4th'], wave=sawtooth_wave, signal_buffer=samples_field, noise_max_amp=0.0, freq=4000
#             )
# )
#
# Model.audio_test(
#     model=tf.keras.models.load_model('models/model_3GZZ6/model_3GZZ6.h5')
# )
