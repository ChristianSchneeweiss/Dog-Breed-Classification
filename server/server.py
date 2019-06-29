import argparse
import logging
import pickle

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session

tf_logger = logging.getLogger("tensorflow")
tf_logger.setLevel(logging.ERROR)

logger = logging.getLogger("server")
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, help="An optional model name argument")

args = parser.parse_args()
model_name = args.model if args.model else "mobilenet"

app = Flask(__name__, static_url_path="/static")

sess = tf.Session()
graph = tf.get_default_graph()

set_session(sess)

model_names = {
    "nasnetmobile": ["nasnetmobile-model-0.81acc-0.70loss-1561201841.hdf5", (224, 224)],
    "nasnetlarge": ["nasnetlarge-model-0.94acc-0.28loss-1561191444.hdf5", (331, 331)],
    "mobilenet": ["mobilenetv2-model-0.73acc-0.92loss-1561064081.hdf5", (224, 224)]
}

if model_name not in model_names.keys():
    logger.critical(f"Error: {model_name} is not available")
    quit(1)

logger.info(f"Chosen model is {model_name}")

model = load_model(f"../models/{model_names[model_name][0]}")
model._make_predict_function()
graph = tf.get_default_graph()

pkl_file = open('../classes.pickle', 'rb')
classes = pickle.load(pkl_file)
pkl_file.close()

classes = {v: k for k, v in classes.items()}


def make_prediction(img):
    global graph
    global sess
    with graph.as_default():
        set_session(sess)
        return model.predict(np.array([img]))


@app.route('/')
def root():
    return app.send_static_file("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    label_predict = "No prediction"
    if request.method == 'POST':
        filestr = request.files['file'].read()
        npimg = np.frombuffer(filestr, np.uint8)
        imageBGR = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        img = preprocess_image(imageBGR)
        label_predict = make_label_prediction(img)
    
    return jsonify({"prediction": label_predict})


def make_label_prediction(img):
    preds = make_prediction(img)
    preds = preds.argmax()
    label_predict = classes[preds]
    return label_predict


def preprocess_image(imageBGR):
    img = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, model_names[model_name][1])
    img = img / 255
    return img


app.run()
