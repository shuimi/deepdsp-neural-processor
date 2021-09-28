from random import uniform, randint, shuffle

from neural_model.dsp_core import *
from neural_model.intervals_config import chords

# amount of buffer samples (raw and processed) to generate
MULTIWAVE_SIGNAL_SAMPLES_AMOUNT = 512
ADDITIVE_SIGNAL_SAMPLES_AMOUNT = 512

# modulators
pulse_wave_modulation = np.linspace(-1, 1, 100)
filter_decay_modulation = np.linspace(0, 1, 100)

# dataset generation
input_signal_samples = []


# mixed waves signal with modulations

for j in range(10):

    low_bound_hz = 50
    hi_bound_hz = 500

    modulation_frequencies_amount = 10

    modulation_frequencies = [
        uniform(low_bound_hz, hi_bound_hz) for k in range(modulation_frequencies_amount)
    ]

    for i in range(MULTIWAVE_SIGNAL_SAMPLES_AMOUNT):
        input_signal_samples += [
            multiwave_oscillator(
                samples_field,
                freq=NYQUIST_FREQUENCY * triangular_wave(
                    i, freq=modulation_frequencies[7] * sine_wave(i, freq=modulation_frequencies[8])
                ),
                sine_amp=triangular_wave(i, freq=modulation_frequencies[0]),
                sine_phase=2 * np.pi * sine_wave(i, freq=modulation_frequencies[1]),
                sawtooth_amp=sine_wave(i, freq=modulation_frequencies[2]),
                sawtooth_phase=2 * np.pi * triangular_wave(i, freq=modulation_frequencies[3]),
                square_amp=triangular_wave(i, freq=modulation_frequencies[4]),
                triangle_amp=sine_wave(i, freq=modulation_frequencies[5]),
                pulse_amp=sawtooth_wave(i, freq=modulation_frequencies[6]),
                pulse_pwm=pulse_wave_modulation[i % 100],
                normalize=False
            )
        ]


# mixed waves signal with modulations and filtering

for i in range(MULTIWAVE_SIGNAL_SAMPLES_AMOUNT):
    input_signal_samples += [
        normalize_filter(
            lowpass(
                multiwave_oscillator(
                    samples_field,
                    freq=NYQUIST_FREQUENCY * sawtooth_wave(i, freq=200 * sine_wave(i, freq=90)),
                    sine_amp=triangular_wave(i, freq=220),
                    sine_phase=2 * np.pi * square_wave(i, freq=220),
                    sawtooth_amp=sine_wave(i, freq=80),
                    sawtooth_phase=2 * np.pi * triangular_wave(i, freq=320),
                    square_amp=triangular_wave(i, freq=55),
                    triangle_amp=sawtooth_wave(i, freq=1110),
                    pulse_amp=sawtooth_wave(i, freq=440),
                    pulse_pwm=pulse_wave_modulation[i % 100],
                    normalize=False
                )
                + white_noise(samples_field, 0.5),
                filter_decay_modulation[i % 100]
            ),
            NORMALIZATION_THRESHOLD
        )
    ]


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


# generate single-wave additive signals with lowpass filter modulation

for wave in [sine_wave, square_wave, sawtooth_wave, pulse_wave, triangular_wave]:
    lf_bound = uniform(20, 50)
    hf_bound = uniform(600, 12000)
    noise_max_amp = uniform(0.2, 1.0)
    harmonics_amount = randint(1, 16)

    for i in range(ADDITIVE_SIGNAL_SAMPLES_AMOUNT):
        input_signal_samples += [
            lowpass(
                rand_additive_signal(samples_field, harmonics_amount, lf_bound, hf_bound, noise_max_amp, wave),
                filter_decay_modulation[i % 100]
            )
        ]


# generate single-wave (chord-based) additive sweep signals with white noise

FRAMES_AMOUNT = 256
NOISE_AMP_CEIL = 0.2
MIN_SIGNAL_AMP = 0.2
AMP_VARIANTS_AMOUNT = 4

for amp_variant in np.linspace(MIN_SIGNAL_AMP, NORMALIZATION_THRESHOLD, AMP_VARIANTS_AMOUNT):
    for chord in chords.values():
        for wave in [sine_wave, square_wave, sawtooth_wave, pulse_wave, triangular_wave]:
            for i in range(FRAMES_AMOUNT):
                input_signal_samples += [
                    normalize_filter(
                        chord_additive_signal(chord, wave, samples_field, 10 + 10 * i, NOISE_AMP_CEIL),
                        amp_variant
                    )
                ]

# generate constant signals with DC offset
DC_STEPS_AMOUNT = 16
for amplitude in np.linspace(-4, 4, DC_STEPS_AMOUNT):
    input_signal_samples += [
        bias_wave(amplitude)
    ]

# make list np.array
shuffle(input_signal_samples)
input_signal_samples = np.asarray(input_signal_samples)


def approximation_target_filter(signal_buffer):
    return lowpass(signal_clipping_filter(signal_buffer, sample_hard_clip), 0.85)


# the processing which will be approximated by model
output_signal_samples = np.array(
    [approximation_target_filter(signal) for signal in input_signal_samples]
)


# data export

np.save('dataset/input_signal_samples.npy', input_signal_samples)
np.save('dataset/output_signal_samples.npy', output_signal_samples)
