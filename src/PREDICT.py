import warnings
import logging
import numpy as np
import librosa as lb
import pandas as pd
# from src.vggish import VGGISH

warnings.filterwarnings("ignore")

# from tensorflow import keras
# from keras.models import Model
# from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D
# from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
# from keras.utils.layer_utils import get_source_inputs
# from keras.engine.topology import get_source_inputs
# from keras import backend as K
###################################################
logger = logging.getLogger('test')
###############################################################
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

n_fft = 2 ** int(np.ceil(np.log(int(round(SAMPLE_RATE*STFT_WINDOW_LENGTH_SECONDS)))/np.log(2.0)))


features_sample_rate = 1.0 / STFT_HOP_LENGTH_SECONDS
example_window_length = int(round(
    EXAMPLE_WINDOW_SECONDS * features_sample_rate))
example_hop_length = int(round(
    EXAMPLE_HOP_SECONDS * features_sample_rate))

def frame(data,window_length,hop_length):
    num_samples = data.shape[0]
    num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
    shape = (1, window_length) + data.shape[1:]
    strides = (data.strides[0] * hop_length,) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

def predict(y,fr,sound_extractor):
    # model = keras.models.load_model("static\src\Emotion.h5")
    # sound_model = VGGISH(load_weights=False,input_shape=(NUM_FRAMES,NUM_BANDS,1))
    # x = sound_model.get_layer(name='full_connect2').output
    # sound_extractor = Model(sound_model.input,x)

    mel_spectrogram = lb.feature.melspectrogram(y,sr=SAMPLE_RATE,
        n_fft= int(n_fft),
        hop_length= round(STFT_HOP_LENGTH_SECONDS*SAMPLE_RATE),
        win_length=round(STFT_WINDOW_LENGTH_SECONDS*SAMPLE_RATE),
        n_mels=NUM_MEL_BINS,
        fmax=MEL_MAX_HZ,
        fmin=MEL_MIN_HZ,
        window='hann'
        )
    mel_spectrogram = mel_spectrogram.T

    log_mel_spectrogram = np.log(mel_spectrogram)

    log_mel_spectrogram = frame(log_mel_spectrogram,window_length=example_window_length,
        hop_length=example_hop_length)
    
    spectrogram = np.expand_dims(log_mel_spectrogram,3)

    features = sound_extractor.predict(spectrogram)

    return features
