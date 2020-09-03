import flask
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import tensorflow as tf
import os
from PIL import Image
import numpy as np
import base64
import io 

MODELS_PATH = './models/'
OUTPUT_PATH = './output/'

BASE_MODEL = 'SRWNNbase.h5'

srwnnModelPaht = MODELS_PATH +  BASE_MODEL

app = Flask(__name__)

def generate(imageInput, modelPath):
    generator = tf.keras.models.load_model(modelPath)

    imageInput = Image.open(imageInput.stream)
    arrayInput = np.array(imageInput)

    input = tf.cast(arrayInput, tf.float32)[...,:3]
    input = (input/127.5) - 1
    image = tf.expand_dims(input, axis = 0)
    
    genOutput = generator(image, training =  False) 

    newImagePath = OUTPUT_PATH + '2xImage.png'

    tf.keras.preprocessing.image.save_img(newImagePath, genOutput[0, ...])
    #reshapeOut = np.array(genOutput[0,...])
    #newImageArray = ((reshapeOut+1)/2)

    #newImage = Image.fromarray(newImageArray, 'RGB')

    return newImagePath

def getModelPath(modelConfig):
    if modelConfig == '0000':
        print('model path: ', srwnnModelPaht)
        return srwnnModelPaht
    if modelConfig != '0000': 
        return srwnnModelPaht 


@app.route('/generate', methods=['POST'])
def gen():
    data = {"success": False}
    if request.files.get("image"):
        image = request.files['image']
        
        payload = request.form.to_dict()
        modelConfig = payload['model']
        print('model config: ', modelConfig)

        modelPathStr = getModelPath(modelConfig)

        newImgPath = generate(image, modelPathStr)

        with open(newImgPath, "rb") as img_file:
            my_string = base64.b64encode(img_file.read())

        img_str = my_string

        os.remove(newImgPath)

        #return jsonify({'status':str(img_base64)})
        data["success"] = True

    return flask.jsonify({'msg': data, 'img': str(img_str) })
    


if __name__ == '__main__':
	app.run(debug = True)
