import os

import cv2
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
from tensorflow.keras.models import load_model


# resizing and converting to RGB
def load_and_preprocess_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


BASEPATH = "../stanford-dogs-dataset/test/"

LABELS = set()

paths = []

for d in os.listdir(BASEPATH):
    LABELS.add(d)
    paths.append((BASEPATH + d, d))

X = []
y = []

for path, label in paths:
    for image_path in os.listdir(path):
        image = load_and_preprocess_image(path + "/" + image_path)
        
        X.append(image)
        y.append(label)

X = np.array(X)
# y = encoder.fit_transform(np.array(y))
y = np.array(y)

print(X[0].dtype)
print(y[0])

print(X.shape)
print(y.shape)
plt.imshow(X[0])

model = load_model("../models/vgg16-model-0.61acc-1.46loss-1560804023.hdf5")

img = X[0]
plt.imshow(img)
predictions = model.predict(np.array([img]))
print(predictions)
