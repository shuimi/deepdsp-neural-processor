from random import uniform

from neural_model.dsp_core.dsp_core import *


# additive signals generation (rand harmonics)

def rand_additive_signal(signal_buffer, harmonics_amount, low_freq_bound, hi_freq_bound, noise_max_amp, wave):
    signal = np.zeros(len(signal_buffer))

    for harmonic in range(harmonics_amount):
        signal = signal + wave(
            signal_buffer,
            freq=uniform(low_freq_bound, hi_freq_bound),
            phase=2 * np.pi * random(),
            amp=uniform(-NORMALIZATION_THRESHOLD, NORMALIZATION_THRESHOLD)
        )

    signal = normalize_filter(signal, NORMALIZATION_THRESHOLD)
    return normalize_filter(signal + white_noise(signal_buffer, uniform(0, noise_max_amp)), NORMALIZATION_THRESHOLD)


# additive signals generation (chord-based)

def chord_additive_signal(chord, wave, signal_buffer, freq, noise_max_amp):
    signal = np.zeros(len(signal_buffer))

    for interval in chord:
        signal = signal + wave(
            signal_buffer,
            freq=freq * interval,
            amp=uniform(-NORMALIZATION_THRESHOLD, NORMALIZATION_THRESHOLD)
        )

    signal = normalize_filter(signal, NORMALIZATION_THRESHOLD)
    return normalize_filter(signal + white_noise(signal_buffer, uniform(0, noise_max_amp)), NORMALIZATION_THRESHOLD)


def approximation_target_filter(signal_buffer):
    return butter_bandpass_filter(signal_clipping_filter(signal_buffer, sample_hard_clip), cutoff_low=560, order=6)
