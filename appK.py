import requests
from PIL import Image
from keras.preprocessing import image
import numpy as np
from io import BytesIO
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Input
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model, Sequential
from keras import optimizers
from keras.models import model_from_json
from keras.datasets import mnist
from keras.layers import Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import sys
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime

app = Flask(__name__)
# graph = tf.get_default_graph()
# input_tensor = Input(shape=(50, 50, 3))
# vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# #ローカルモデルを構築
# top_model = Sequential()
# top_model.add(Flatten(input_shape=vgg16.output_shape[1:])) 
# top_model.add(Dense(256, activation='relu'))
# top_model.add(Dropout(0.5))
# top_model.add(Dense(2, activation='softmax'))


# # VGG16とFCを接続
# model = Model(input=vgg16.input, output=top_model(vgg16.output))
# model.load_weights('./weight/my_model_weight.h5')

# # コンパイルする
# model.compile(loss='categorical_crossentropy',
#               optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
#               metrics=['accuracy'])





# flaskの起動
@app.route('/', methods = ['GET', 'POST']) 
def upload_file():
    if request.method == 'GET':
        return render_template('indexK.html')
    elif request.method == 'POST':
        # f = request.files['file']
        # form_img = Image.open(f)  
    
        # img_rev = form_img.resize((50, 50)) # 画像をリサイズ
        # img_rev=img_rev.convert('RGB')    
        # x = image.img_to_array(img_rev) 
        # x = np.expand_dims(x, axis=0)  
        # x = x / 255.0

        # global graph
        #スレッド間を共有するため、graphを開きます
        # # with graph.as_default():
        #     score = model.predict(x)
        #     pred = np.argmax(score)
        #     animal = ['犬','猫']
        
        #index.htmlに値を渡す
        return render_template('resultK.html',pred_animal = 1, score=60) 

@app.route('/resultK')
def result():
    return render_template('resultK.html')

if __name__ == '__main__':
    #app.debug = True 
    app.run(host='0.0.0.0')
    #app.run()