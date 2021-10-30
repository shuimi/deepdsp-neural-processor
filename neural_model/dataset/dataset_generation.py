from itertools import combinations
from random import uniform, randint, shuffle, choices, getrandbits

from scipy.signal import convolve

from neural_model.config.intervals_config import chords
from neural_model.dataset.generation_tools import *


def log(step, data):
    samples_peaks = np.array([np.max(np.abs(sample)) for sample in data])
    samples_rms = np.array([rms(sample) for sample in data], dtype='double')

    mean_rms = np.mean(samples_rms)
    mean_peak = np.mean(samples_peaks)

    print('[' + str(step) + ', mean_rms = ' + str(mean_rms) + ', mean_peak = ' + str(mean_peak) + ']')


# amount of buffer samples (raw and processed) to generate
MULTIWAVE_SIGNAL_SAMPLES_AMOUNT = 2048
ADDITIVE_SIGNAL_SAMPLES_AMOUNT = 2048

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
                normalize=True
            )
        ]

log('Mixed waves, modulations', input_signal_samples)


# mixed waves signal with modulations and filtering

for i in range(MULTIWAVE_SIGNAL_SAMPLES_AMOUNT):
    input_signal_samples += [
        normalize_filter(
            butter_lowpass_filter(
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
                filter_decay_modulation[i % 100] * 10000 + 500
            ),
            NORMALIZATION_THRESHOLD
        )
    ]

log('Mixed waves, modulations, lowpass', input_signal_samples)


# generate single-wave additive signals with lowpass filter modulation

for wave in [sine_wave, square_wave, sawtooth_wave, pulse_wave, triangular_wave]:
    lf_bound = uniform(20, 50)
    hf_bound = uniform(600, 12000)
    noise_max_amp = uniform(0.2, 1.0)
    harmonics_amount = randint(1, 16)

    for i in range(ADDITIVE_SIGNAL_SAMPLES_AMOUNT):
        input_signal_samples += [
            normalize_filter(
                butter_lowpass_filter(
                    rand_additive_signal(samples_field, harmonics_amount, lf_bound, hf_bound, noise_max_amp, wave),
                    filter_decay_modulation[i % 100] * 10000 + 500
                ),
                NORMALIZATION_THRESHOLD
            )
        ]

log('Additive, rand harmonics, modulation, lowpass', input_signal_samples)


# generate single-wave (chord-based) additive sweep signals with white noise

FRAMES_AMOUNT = 256
NOISE_AMP_CEIL = 0.25

for chord in chords.values():
    for wave in [sine_wave, square_wave, sawtooth_wave, pulse_wave, triangular_wave]:
        for i in range(FRAMES_AMOUNT):
            input_signal_samples += [
                normalize_filter(
                    chord_additive_signal(chord, wave, samples_field, 10 + 10 * i, NOISE_AMP_CEIL),
                    NORMALIZATION_THRESHOLD
                )
            ]

log('Single-wave (chord-based) additive sweep signals with white noise', input_signal_samples)


# detuned signals generation
DETUNE_SELECTION_SIZE = 6144

for wave in [sine_wave, square_wave, sawtooth_wave, pulse_wave, triangular_wave]:
    input_signal_samples += [
        normalize_filter(
            detune(
                samples_field,
                wave,
                freq=uniform(0, NYQUIST_FREQUENCY / 6),
                voices_amount=randint(2, 32),
                detune_st=uniform(0.01, 1.0),
                blend=uniform(0.5, 1.0)
            ),
            NORMALIZATION_THRESHOLD
        )
    ]

log('detuned signals', input_signal_samples)


# generate constant signals with DC offset
DC_STEPS_AMOUNT = 1024
for amplitude in np.linspace(-NORMALIZATION_THRESHOLD, NORMALIZATION_THRESHOLD, DC_STEPS_AMOUNT):
    input_signal_samples += [
        bias_wave(amplitude)
    ]

log('constant signals with DC offset', input_signal_samples)


# dataset augmentation by random samples normalized convolution
SELECTION_ITERATIONS = 8
SELECTION_SIZE = 32  # COMBINATIONS_AMOUNT = SELECTION_SIZE! / (2! * (SELECTION_SIZE - 2)!)

for _ in range(SELECTION_ITERATIONS):
    combs = combinations(choices(input_signal_samples, k=SELECTION_SIZE), 2)

    for sample_a, sample_b in combs:
        input_signal_samples += [
            normalize_filter(
                convolve(sample_a, sample_b)[0:BUFFER_SIZE],
                NORMALIZATION_THRESHOLD
            )
        ]

log('augmentation by random samples normalized convolution', input_signal_samples)


# random cut augmentation
def rand_cut(buffer, order=1):
    new_buffer = buffer

    for _ in range(order):
        left_bound = randint(0, BUFFER_SIZE - 2)
        right_bound = randint(left_bound + 1, BUFFER_SIZE - 1)

        for index in range(left_bound, right_bound, 1):
            new_buffer[index] = 0

    return new_buffer


RANDOM_CUT_SELECTION_SIZE = 4096

for signal in choices(input_signal_samples, k=RANDOM_CUT_SELECTION_SIZE):
    input_signal_samples += [rand_cut(signal, order=randint(1, 3))]

log('random cut augmentation', input_signal_samples)


# dataset augmentation on phase
PHASE_STEPS = 1024

for signal in choices(input_signal_samples, k=PHASE_STEPS):
    phase_bias = randint(0, BUFFER_SIZE / 2 - 1)
    input_signal_samples += [np.roll(signal, phase_bias)]

log('dataset augmentation on phase', input_signal_samples)


# dataset augmentation on phase
FLIP_STEPS = 1024

for signal in choices(input_signal_samples, k=FLIP_STEPS):
    input_signal_samples += [np.flip(signal)]

log('dataset augmentation on phase', input_signal_samples)


# dataset augmentation by lowpass filter
LOWPASS_FILTER_STEPS = 1024
lowpass_augmentation_dropped = 0

for signal in choices(input_signal_samples, k=LOWPASS_FILTER_STEPS):

    _signal = normalize_filter(
        butter_lowpass_filter(signal, cutoff=uniform(20, 20000), order=randint(1, 10), analog=bool(getrandbits(1)))
        , NORMALIZATION_THRESHOLD
    )

    if rms(_signal) > 0.5:
        input_signal_samples += [
            _signal
        ]
    else:
        lowpass_augmentation_dropped += 1

print(
    '-lowpass augmentation: [ added: ' +
    str(LOWPASS_FILTER_STEPS - lowpass_augmentation_dropped) +
    ', dropped: ' +
    str(lowpass_augmentation_dropped) + ']'
)
log('lowpass augmentation', input_signal_samples)


# dataset augmentation by highpass filter
HIGHPASS_FILTER_STEPS = 2048
highpass_augmentation_dropped = 0

for signal in choices(input_signal_samples, k=HIGHPASS_FILTER_STEPS):

    _signal = normalize_filter(
        butter_highpass_filter(signal, cutoff=uniform(20, 20000), order=randint(1, 10), analog=bool(getrandbits(1)))
        , NORMALIZATION_THRESHOLD
    )

    if rms(_signal) > 0.5:
        input_signal_samples += [
            _signal
        ]
    else:
        highpass_augmentation_dropped += 1

print(
    '-highpass augmentation: [ added: ' +
    str(HIGHPASS_FILTER_STEPS - highpass_augmentation_dropped) +
    ', dropped: ' +
    str(highpass_augmentation_dropped) + ']'
)
log('highpass augmentation', input_signal_samples)


# dataset augmentation by bandpass filter
BANDPASS_FILTER_STEPS = 4096
bandpass_augmentation_dropped = 0

for signal in choices(input_signal_samples, k=BANDPASS_FILTER_STEPS):

    cutoff_low = uniform(200, 10000)
    cutoff_high = uniform(cutoff_low, 20000)

    _signal = normalize_filter(
            butter_bandpass_filter(
                signal,
                cutoff_low=cutoff_low,
                cutoff_hi=cutoff_high,
                order=randint(1, 10),
                analog=bool(getrandbits(1))
            ),
            NORMALIZATION_THRESHOLD
        )

    if rms(_signal) > 0.5:
        input_signal_samples += [
            _signal
        ]
    else:
        bandpass_augmentation_dropped += 1

print(
    '-bandpass augmentation: [ added: ' +
    str(BANDPASS_FILTER_STEPS - bandpass_augmentation_dropped) +
    ', dropped: ' +
    str(bandpass_augmentation_dropped) + ']'
)
log('bandpass augmentation', input_signal_samples)


# make list np.array
shuffle(input_signal_samples)
input_signal_samples = np.asarray(input_signal_samples)


# the processing which will be approximated by model
output_signal_samples = np.array(
    [approximation_target_filter(signal) for signal in input_signal_samples]
)

# data export

np.save('input_signal_samples.npy', input_signal_samples)
np.save('output_signal_samples.npy', output_signal_samples)
