import os
import uuid
from flask import Flask, flash, jsonify, render_template, request, redirect, url_for
import librosa as lb
import numpy as np
from src import PREDICT, vggish
from src.settings import *

import keras
from keras.models import Model


UPLOAD_FOLDER = 'static/files'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET','POST'])
def root():
    return render_template("record.html")


@app.route('/predict')
def predict():

    y,fr = lb.load('static/files/recordedAudio.wav')

    sound_model = vggish.VGGISH(load_weights=False,input_shape=(NUM_FRAMES,NUM_BANDS,1))
    x = sound_model.get_layer(name='full_connect2').output
    sound_extractor = Model(sound_model.input,x)
    model = keras.models.load_model("src\Emotion32.h5")
    
    features = PREDICT.predict(y,fr,sound_extractor)
    result = model.predict(features)

    emotionPool = ['Angry','Happy','Neutral','Sad','Surprise']
    
    emotion = emotionPool[np.argmax(result[0])]
    print(result[0])
    return render_template("predict.html",content=emotion)



@app.route('/save-record', methods=['POST'])
def save_record():
    # check if the post request has the file part
    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    print(type(file))
   
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
      
    file_name = "recordedAudio.wav"  
    full_file_name = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    file.save(full_file_name)


    return '<h1>Success</h1>'   


if __name__ == '__main__':
    app.run(debug=True)