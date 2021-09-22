# config
SAMPLE_RATE = int(44100)

# buffer settings
BUFFER_SIZE = int(64)
BUFFER_SAMPLE_INDEXING_STARTING_POINT = int(0)
BUFFER_SAMPLE_INDEXING_STEP = int(1)

# amount of buffer samples (raw and processed) to generate
SIGNAL_SAMPLES_AMOUNT = 512

#
NORMALIZATION_THRESHOLD = float(4)  # threshold at which the signal in each buffer is normalized
AUGMENTATION_QUANTITY = 4

# generator config
OUTER_SPHERE_RADIUS = sum([NORMALIZATION_THRESHOLD ** 2 for i in range(BUFFER_SIZE)]) ** 0.5  # euclidean distance

# model config
BATCH_SIZE = 32
EPOCHS_AMOUNT = 65536

EARLY_STOPPING_PATIENCE = 4
EARLY_STOPPING_MIN_DELTA = 0.0001
