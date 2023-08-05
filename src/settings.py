NUM_FRAMES = 256  # Frames in input mel-spectrogram patch.
NUM_BANDS = 32  # Frequency bands in input mel-spectrogram patch.
EMBEDDING_SIZE = 128  # Size of embedding layer.

# Hyperparameters used in feature and example generation.
SAMPLE_RATE = 16000
STFT_WINDOW_LENGTH_SECONDS = 0.025
STFT_HOP_LENGTH_SECONDS = 0.010
NUM_MEL_BINS = NUM_BANDS
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
EXAMPLE_WINDOW_SECONDS = 2.56  # Each example contains 96 10ms frames
EXAMPLE_HOP_SECONDS = 2.56 

# n_fft = 2 ** int(np.ceil(np.log(int(round(SAMPLE_RATE*STFT_WINDOW_LENGTH_SECONDS)))/np.log(2.0)))


# features_sample_rate = 1.0 / STFT_HOP_LENGTH_SECONDS
# example_window_length = int(round(
#     EXAMPLE_WINDOW_SECONDS * features_sample_rate))
# example_hop_length = int(round(
#     EXAMPLE_HOP_SECONDS * features_sample_rate))