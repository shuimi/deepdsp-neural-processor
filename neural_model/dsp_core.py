import numpy as np
from random import random
from neural_model.config import *

# config
BASE_OSC_FREQUENCY = 440.0
BASE_OSC_AMP = 1.0
BASE_OSC_PHASE = 0.0

# samples field
samples_field = np.arange(BUFFER_SAMPLE_INDEXING_STARTING_POINT, BUFFER_SIZE, BUFFER_SAMPLE_INDEXING_STEP)


# oscillators

def bias_wave(value=0.0):
    return np.zeros(BUFFER_SIZE) + value


def white_noise(buffer_indexes, max_amp=1.0):
    new_buffer = []
    for _ in buffer_indexes:
        new_buffer += [max_amp * (random() - 0.5) * 2]
    return np.array(new_buffer)


def sine_wave(buffer_indexes, freq=BASE_OSC_FREQUENCY, phase=BASE_OSC_PHASE, amp=BASE_OSC_AMP):
    angular_freq = 2 * np.pi * freq
    if angular_freq == 0:
        period = float('inf')
    else:
        period = SAMPLE_RATE / angular_freq
    return amp * np.sin(buffer_indexes / period + phase)


def triangular_wave(buffer_indexes, freq=BASE_OSC_FREQUENCY, phase=BASE_OSC_PHASE, amp=BASE_OSC_AMP):
    if freq == 0:
        period = float('inf')
    else:
        period = SAMPLE_RATE / freq
    return amp * (2 / np.pi) * np.arcsin(np.sin((2 * np.pi * buffer_indexes) / period + phase))


def sawtooth_wave(buffer_indexes, freq=BASE_OSC_FREQUENCY, phase=BASE_OSC_PHASE, amp=BASE_OSC_AMP):
    if freq == 0:
        period = float('inf')
    else:
        period = SAMPLE_RATE / freq

    def transform(t):
        return 2 * (t / period - np.floor(0.5 + t / period + phase) + phase)

    if buffer_indexes is float or int:
        return transform(buffer_indexes)

    processed_buffer = []
    for sample in buffer_indexes:
        processed_buffer += [transform(sample)]

    return np.multiply(np.array(processed_buffer), amp)


def pulse_wave(buffer_indexes, freq=BASE_OSC_FREQUENCY, phase=BASE_OSC_PHASE, amp=BASE_OSC_AMP, pwm=0.5):
    if not -1.0 <= pwm <= 1.0:
        raise Exception('pulse_wave error: pwm parameter must be in range -1.0 .. 1.0')

    def adjustable_sgn(x, threshold):
        if x > threshold:
            return 1
        elif x < threshold:
            return -1
        else:
            return 0

    def crop(buffer, threshold):
        new_buffer = []
        for sample in buffer:
            new_buffer += [adjustable_sgn(sample, threshold)]
        return np.array(new_buffer)

    return np.multiply(crop(sawtooth_wave(buffer_indexes, freq=freq, phase=phase, amp=1.0), pwm), amp)


def square_wave(buffer_indexes, freq=BASE_OSC_FREQUENCY, phase=BASE_OSC_PHASE, amp=BASE_OSC_AMP):
    if freq == 0:
        period = float('inf')
    else:
        period = SAMPLE_RATE / freq
    return np.multiply(np.sign(np.sin(2 * np.pi * buffer_indexes / period + phase)), amp)


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


# IIR lowpass filter
def lowpass_iir_filter():
    previous_sample = 0.0

    def lowpass_filter(x, decay):
        b = 1.0 - decay
        nonlocal previous_sample
        previous_sample += b * (x - previous_sample)
        return previous_sample

    return lowpass_filter


def lowpass(buffer, decay):
    lpf = lowpass_iir_filter()

    processed_buffer = []
    for sample in buffer:
        processed_buffer += [lpf(sample, decay)]

    return processed_buffer


# oscillators

def multiwave_oscillator(
        buffer_indexes,

        freq=BASE_OSC_FREQUENCY,
        max_amp=NORMALIZATION_THRESHOLD,

        sine_phase=BASE_OSC_PHASE,
        sine_amp=BASE_OSC_AMP,

        triangle_phase=BASE_OSC_PHASE,
        triangle_amp=BASE_OSC_AMP,

        sawtooth_phase=BASE_OSC_PHASE,
        sawtooth_amp=BASE_OSC_AMP,

        square_phase=BASE_OSC_PHASE,
        square_amp=BASE_OSC_AMP,

        pulse_phase=BASE_OSC_PHASE,
        pulse_amp=BASE_OSC_AMP,
        pulse_pwm=0.5,

        normalize=False

):
    waves_combination = sine_wave(buffer_indexes, freq=freq, phase=sine_phase, amp=sine_amp) + \
        triangular_wave(buffer_indexes, freq=freq, phase=triangle_phase, amp=triangle_amp) + \
        sawtooth_wave(buffer_indexes, freq=freq, phase=sawtooth_phase, amp=sawtooth_amp) + \
        square_wave(buffer_indexes, freq=freq, phase=square_phase, amp=square_amp) + \
        pulse_wave(buffer_indexes, freq=freq, phase=pulse_phase, amp=pulse_amp, pwm=pulse_pwm)

    if not normalize:
        return waves_combination
    return normalize_filter(waves_combination, max_amp=max_amp)
