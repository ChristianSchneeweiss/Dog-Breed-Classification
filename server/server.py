import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, static_url_path="/static")

sess = tf.Session()
graph = tf.get_default_graph()

set_session(sess)

model = load_model("../models/vgg16-model-0.61acc-1.46loss-1560804023.hdf5")
model._make_predict_function()
graph = tf.get_default_graph()

pkl_file = open('../vgg16_encoder.pkl', 'rb')
encoder = pickle.load(pkl_file)
pkl_file.close()


def make_prediction(img):
    global graph
    global sess
    with graph.as_default():
        set_session(sess)
        return model.predict(np.array([img]))


@app.route('/')
def root():
    # return "Hello World"
    return app.send_static_file("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    label_predicts = "No prediction"
    if request.method == 'POST':
        filestr = request.files['file'].read()
        npimg = np.frombuffer(filestr, np.uint8)
        imageBGR = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        preds = make_prediction(img)
        preds = preds.argmax()
        label_predicts = encoder.inverse_transform(np.array([preds]))
    
    return jsonify({"prediction": label_predicts[0]})


app.run()
