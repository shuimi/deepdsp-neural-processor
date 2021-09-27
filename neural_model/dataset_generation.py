from random import uniform, randint

from neural_model.dsp_core import *

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


# additive sine wave random signals generation

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


for i in range(ADDITIVE_SIGNAL_SAMPLES_AMOUNT):
    input_signal_samples += [
        lowpass(
            rand_additive_signal(samples_field, randint(1, 16), 50, 8000, 1.0, sine_wave),
            filter_decay_modulation[i % 100]
        ),
        lowpass(
            rand_additive_signal(samples_field, randint(1, 16), 50, 600, 0.8, square_wave),
            filter_decay_modulation[i % 100]
        ),
        lowpass(
            rand_additive_signal(samples_field, randint(1, 16), 50, 8000, 1.2, triangular_wave),
            filter_decay_modulation[i % 100]
        ),
        lowpass(
            rand_additive_signal(samples_field, randint(1, 16), 50, 8000, 1.4, pulse_wave),
            filter_decay_modulation[i % 100]
        ),
        lowpass(
            rand_additive_signal(samples_field, randint(1, 16), 20, 900, 0.7, sawtooth_wave),
            filter_decay_modulation[i % 100]
        )
    ]


input_signal_samples = np.asarray(input_signal_samples)


# the processing which will be approximated by model
output_signal_samples = np.array(
    [signal_clipping_filter(signal, sample_hard_clip) for signal in input_signal_samples]
)


# data export

# TODO: export generated data
