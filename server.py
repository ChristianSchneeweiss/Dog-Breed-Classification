from flask import Flask, request
import matplotlib.pyplot as plt
import numpy as np
import cv2

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        filestr = request.files['file'].read()
        # convert string data to numpy array
        npimg = np.frombuffer(filestr, np.uint8)
        img = cv2.imdecode(npimg, cv2.CV_LOAD_IMAGE_UNCHANGED)
        plt.imshow(img)
        plt.show()


app.run()
