# config
SAMPLE_RATE = int(44100)
NYQUIST_FREQUENCY = SAMPLE_RATE // 2

# buffer settings
BUFFER_SIZE = int(512)
BUFFER_SAMPLE_INDEXING_STARTING_POINT = int(0)
BUFFER_SAMPLE_INDEXING_STEP = int(1)

# data generator config
NORMALIZATION_THRESHOLD = float(4)  # threshold at which the signal in each buffer is normalized

# model config
BATCH_SIZE = 32
EPOCHS_AMOUNT = 400
