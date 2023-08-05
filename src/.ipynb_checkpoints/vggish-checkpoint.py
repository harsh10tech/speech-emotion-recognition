import logging
import os
import numpy as np
import warnings

from tensorflow import keras
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.utils.layer_utils import get_source_inputs
from keras import backend as K

###################################################
logger = logging.getLogger('test')
###############################################################
NUM_FRAMES = 256  # Frames in input mel-spectrogram patch.
NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
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

features_sample_rate = 1.0 / STFT_HOP_LENGTH_SECONDS
example_window_length = int(round(
    EXAMPLE_WINDOW_SECONDS * features_sample_rate))
example_hop_length = int(round(
    EXAMPLE_HOP_SECONDS * features_sample_rate))

def VGGISH(load_weights= True,weights ='audioset',
    input_tensor=None,input_shape=None,out_dim = None,
    include_top=True,pooling='avg'):

    if weights not in {'audioset',None}:
        raise ValueError('The weights argument should be either None or audioset')
    
    if out_dim is None:
        out_dim = 128
    
    if input_shape is None:
        input_shape = (NUM_FRAMES,NUM_BANDS,1)
    
    if input_tensor is None:
        aud_input = Input(shape=input_shape,name='input_1')
        
    else:
        if not K.is_keras_tensot(input_tensor):
            aud_input = Input(tensor=input_tensor,shape = input_shape,name = 'input_1')
        else:
            aud_input = input_tensor
    
    x = Conv2D(64,(3,3),strides=(1,1), activation='relu',padding='same',name='conv1')(aud_input)
    x = MaxPooling2D((2,2),strides=(2,2),padding='same',name='pool1')(x)

    x = Conv2D(128,(3,3),strides=(1,1),activation='relu',padding='same',name='conv2')(x)
    x = MaxPooling2D((2,2),strides=(2,2),padding='same',name='pool2')(x)

    x = Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',name='conv3.1')(x)
    x = Conv2D(256,(3,3),strides=(1,1),activation='relu',padding='same',name='conv3.2')(x)
    x = MaxPooling2D((2,2),strides=(2,2),padding='same',name='pool3')(x)

    x = Conv2D(512,(3,3),strides=(1,1),activation='relu',padding='same',name='conv4.1')(x)
    x = Conv2D(512,(3,3),strides=(1,1),activation='relu',padding='same',name='conv4.2')(x)
    x = MaxPooling2D((2,2),strides=(2,2),padding='same',name='pool4')(x)
    
    x = Flatten(name='Flatten_')(x)
    x = Dense(4096,activation='relu',name='full_connect1.1')(x)
    x = Dense(4096,activation='relu',name='full_connect1.2')(x)
    x = Dense(out_dim,activation='relu',name='full_connect2')(x)


    # if include_top:
    #     x = Flatten(name='Flatten_')(x)
    #     x = Dense(4096,activation='relu',name='full_connect1.1')(x)
    #     x = Dense(4096,activation='relu',name='full_connect1.2')(x)
    #     x = Dense(out_dim,activation='relu',name='full_connect2')(x)
    # else:
    #     if pooling == 'avg':
    #         x = GlobalAveragePooling2D()(x)
    #     elif pooling == 'max':
    #         x = GlobalMaxPooling2D()(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = aud_input
    
    model = Model(inputs, x , name='VGGish')

    # if load_weights:
    #     if weights == 'audioset':
    #         if include_top:
    #             model.load_weights(WEIGHTS_PATH)
    #         else:
    #             model.load_weights(WEIGHTS_PATH)
    #     else:
    #         print('Failed to load weights')

    return model
