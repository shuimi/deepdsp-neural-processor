import numpy as np
from random import random
from math import floor
from neural_model.config \
    import BUFFER_SIZE, BUFFER_SAMPLE_INDEXING_STEP, BUFFER_SAMPLE_INDEXING_STARTING_POINT, SAMPLE_RATE


# samples field
samples_field = np.arange(BUFFER_SAMPLE_INDEXING_STARTING_POINT, BUFFER_SIZE, BUFFER_SAMPLE_INDEXING_STEP)


# oscillators

def white_noise(max_amp):
    return max_amp * random()


def sine_wave(sample_index, freq=1.0, phase=0.0, amp=1.0):

    period = SAMPLE_RATE / freq

    return amp * np.sin(sample_index / period + phase)


def triangular_wave(sample_index, freq=1.0, phase=0.0, amp=1.0):
    return amp * (2 / np.pi) * np.arcsin(np.sin(freq * sample_index + phase))


def sawtooth_wave(buffer_indexes, freq=1.0, phase=0.0, amp=1.0):

    period = SAMPLE_RATE / freq

    def process_sample(sample_index):
        return -((2 * amp) / np.pi) * np.arctan(1 / np.tan((np.pi * sample_index) / period))

    processed_buffer = []
    for sample in buffer_indexes:
        processed_buffer += [process_sample(sample)]

    return np.multiply(np.array(processed_buffer), amp)


# sample processing functions

def sample_soft_clip(sample_value):
    if sample_value >= 1.0:
        return 1.0
    elif sample_value <= -1.0:
        return -1.0
    else:
        return (3 / 2) * sample_value * (1 - sample_value ** 2 / 3)


def sample_hard_clip(sample_value):
    if sample_value >= 1.0:
        return 1.0
    elif sample_value <= -1.0:
        return -1.0
    else:
        return sample_value


# filters

def signal_clipping_filter(signal_buffer, clipper_function):
    result_buffer = []

    for sample_value in signal_buffer:
        result_buffer = result_buffer + [clipper_function(sample_value)]

    return np.array(result_buffer)


def reverse_polarity_filter(signal_buffer):
    return np.multiply(signal_buffer, -1)


def normalize_filter(signal_buffer, max_amp):
    current_buffer_max_sample_amp = 0.0

    for sample_value in signal_buffer:
        if abs(sample_value) > current_buffer_max_sample_amp:
            current_buffer_max_sample_amp = abs(sample_value)

    if current_buffer_max_sample_amp <= 1:
        return signal_buffer

    normalization_coefficient = max_amp / current_buffer_max_sample_amp

    return np.multiply(signal_buffer, normalization_coefficient)
