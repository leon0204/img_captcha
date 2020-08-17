import base64

import numpy as np
import tensorflow as tf

from io import BytesIO
from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

CAPTCHA_CHARSET = NUMBER  # 验证码字符集
CAPTCHA_LEN = 5  # 验证码长度
CAPTCHA_HEIGHT = 50  # 验证码高度
CAPTCHA_WIDTH = 200  # 验证码宽度

# 10 个 Epochs 训练的模型 rmsprop  adam
MODEL_FILE = './model/train_demo/captcha_adam_binary_crossentropy_bs_700_epochs_50.h5'


def vec2text(vector):
    if not isinstance(vector, np.ndarray):
        vector = np.asarray(vector)
    vector = np.reshape(vector, [CAPTCHA_LEN, -1])
    text = ''
    for item in vector:
        text += CAPTCHA_CHARSET[np.argmax(item)]
    return text


def rgb2gray(img):
    # Y' = 0.299 R + 0.587 G + 0.114 B 
    # https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])


app = Flask(__name__)  # 创建 Flask 实例


# 测试 URL
@app.route('/ping', methods=['GET', 'POST'])
def hello_world():
    return 'pong'


# 验证码识别 URL
@app.route('/predict', methods=['POST'])
def predict():
    response = {'success': False, 'prediction': '', 'debug': 'error'}
    received_image = False
    if request.method == 'POST':
        if request.files.get('image'):  # 图像文件
            image = request.files['image'].read()
            received_image = True
            response['debug'] = 'get image'
        elif request.get_json():  # base64 编码的图像文件
            encoded_image = request.get_json()['image']
            image = base64.b64decode(encoded_image)
            received_image = True
            response['debug'] = 'get json'
        else:
            response['debug'] = 'no one'
        if received_image:
            image = np.array(Image.open(BytesIO(image)))
            image = rgb2gray(image).reshape(1, 50, 200, 1).astype('float32') / 255
            with graph.as_default():
                pred = model.predict(image)
            response['prediction'] = response['prediction'] + vec2text(pred)
            response['success'] = True
            response['debug'] = 'predicted'
    else:
        response['debug'] = 'No Post'
    return jsonify(response)


model = load_model(MODEL_FILE)  # 加载模型
graph = tf.get_default_graph()  # 获取 TensorFlow 默认数据流图

# curl 127.0.0.1:5000/ping
# curl -X POST -F image=@/root/Workspace/leon/test-data/56497.jpg 'http://localhost:5000/predict'
