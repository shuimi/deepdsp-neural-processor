import matplotlib.pyplot as plt
from random import uniform, randint
from neural_model.config import *
from neural_model.dsp_core import *


# random signal generation

def get_rand_hirange_signal(sample_index, harmonics_amount):
    signal = np.array(sine_wave(sample_index, amp=0))

    for harmonic in range(harmonics_amount):
        np.add(
            signal,
            sine_wave(sample_index, freq=uniform(200, 880), phase=np.pi * random(), amp=uniform(0, 4)),
            out=signal
        )

    np.add(signal, white_noise(uniform(0, 4)))

    return normalize_filter(signal, NORMALIZATION_THRESHOLD)


def get_rand_midrange_signal(sample_index, harmonics_amount):
    signal = np.array(sine_wave(sample_index, amp=0))

    for harmonic in range(harmonics_amount):
        np.add(
            signal,
            sine_wave(sample_index, freq=uniform(1, 200), phase=np.pi * random(), amp=uniform(0, 4)),
            out=signal
        )

    np.add(signal, white_noise(uniform(0, 2)))

    return normalize_filter(signal, NORMALIZATION_THRESHOLD)


def get_rand_lowrange_signal(sample_index, harmonics_amount):
    signal = np.array(sine_wave(sample_index, amp=0))

    for harmonic in range(harmonics_amount):
        np.add(
            signal,
            sine_wave(sample_index, freq=uniform(0.01, 1), phase=np.pi * random(), amp=uniform(0, 2)),
            out=signal
        )

    np.add(signal, white_noise(uniform(0, 1)))

    return normalize_filter(signal, NORMALIZATION_THRESHOLD)


input_signal_samples = []


input_signal_samples = np.asarray(input_signal_samples)


# dataset generation

input_signal_samples = []

for sample in range(SIGNAL_SAMPLES_AMOUNT):
    buffer = get_rand_hirange_signal(samples_field, randint(1, 32))
    buffer_normalized = normalize_filter(buffer, 1)

    # augmentation: amplitude
    for i in range(AUGMENTATION_QUANTITY):
        input_signal_samples += [
            np.multiply(buffer_normalized, uniform(-NORMALIZATION_THRESHOLD, NORMALIZATION_THRESHOLD))
        ]

    buffer = get_rand_midrange_signal(samples_field, randint(1, 32))
    buffer_normalized = normalize_filter(buffer, 1)

    # augmentation: amplitude
    for i in range(AUGMENTATION_QUANTITY):
        input_signal_samples += [
            np.multiply(buffer_normalized, uniform(-NORMALIZATION_THRESHOLD, NORMALIZATION_THRESHOLD))
        ]

    buffer = get_rand_lowrange_signal(samples_field, randint(1, 3))
    buffer_normalized = normalize_filter(buffer, 1)

    # augmentation: amplitude
    for i in range(AUGMENTATION_QUANTITY):
        input_signal_samples += [
            np.multiply(buffer_normalized, uniform(-NORMALIZATION_THRESHOLD, NORMALIZATION_THRESHOLD))
        ]


input_signal_samples = np.asarray(input_signal_samples)


# adding sine sweep

SWEEP_LENGTH_SAMPLES = 1024


def sine_sweep(sample_index, base_freq=1, phase=0, amp=1):
    return amp * np.sin((8 * SWEEP_LENGTH_SAMPLES * base_freq) / (sample_index + 1) + phase)


t2 = np.arange(0, SWEEP_LENGTH_SAMPLES, 1)
full_sweep = sine_sweep(t2, base_freq=1, amp=1)
full_sweep[0] = 0

for chunk in np.array_split(full_sweep, SWEEP_LENGTH_SAMPLES / BUFFER_SIZE):
    for amp_coefficient in [0.25, 1, -2]:
        np.append(input_signal_samples, [np.multiply(chunk, amp_coefficient)])
    for amp_coefficient in [4, 32]:
        np.append(input_signal_samples,
                  [np.multiply(
                      signal_clipping_filter(np.multiply(chunk, amp_coefficient), sample_hard_clip),
                      2
                  )])


# some stats about dataset
#
# # distance between two vectors
# def distance(vec1, vec2):
#     _sum = 0.0
#     for index in range(len(vec1)):
#         _sum += abs(vec1[index] - vec2[index]) ** 2
#     return _sum ** 0.5
#
#
# def get_stats():
#     min_distance = distance(input_signal_samples[0], input_signal_samples[1])
#     max_distance = 0
#
#     distances_sum = 0.0
#     iterations = 0
#     for i in range(len(input_signal_samples)):
#         for j in range(i + 1, len(input_signal_samples)):
#             iterations = iterations + 1
#             distances_sum = distances_sum + distance(input_signal_samples[i], input_signal_samples[j])
#             if distances_sum > max_distance:
#                 max_distance = distances_sum
#             if distances_sum < min_distance:
#                 min_distance = distances_sum
#
#     average_distance = distances_sum / iterations
#
#     return {'avg': average_distance, 'min': min_distance, 'max': max_distance}
#
#
# def report_distance_metrics(average_distance, min_distance, max_distance):
#     print('Average distance between samples: ', average_distance)
#     print('Min distance between samples: ', min_distance)
#     print('Max distance between samples: ', max_distance)
#
#
# print('Input samples amount: ', len(input_signal_samples))
# stats = get_stats()
# report_distance_metrics(stats['avg'], stats['min'], stats['max'])


# original signal processing

output_signal_samples = np.array([signal_clipping_filter(signal, sample_hard_clip) for signal in input_signal_samples])

# # dataset plotting
#
# for signal_index in range(len(input_signal_samples)):
#     fig, ax = plt.subplots()
#
#     ax.plot(samples_field, input_signal_samples[signal_index])
#     ax.plot(samples_field, output_signal_samples[signal_index])
#
#     ax.set(xlabel='buffer samples', ylabel='amplitude', title='Signal buffer')
#     ax.grid()
#     plt.show()

# # data export
#
# with open('input_signal_samples.json', 'w') as file:
#     json.dump(input_signal_samples, file)
#
# with open('output_signal_samples.json', 'w') as file:
#     json.dump(output_signal_samples, file)
