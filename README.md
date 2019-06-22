# Dog-Breed-Classification

I found [this](https://www.kaggle.com/jessicali9530/stanford-dogs-dataset) Dataset on Kaggle on wanted to try how good I can train a Deep Learning model on it. The Kaggle Kernel I created - [kernel](https://www.kaggle.com/waterchiller/vgg16-classification-dog-breed)

1. NASNetLarge 94%
2. NASNetMobile 81%
3. InceptionV3 77%
4. EfficientNetB3 76%
5. MobileNetV2 73% (alpha=1.0)
6. EfficientNetB0 68%
7. VGG16 61%

I expected VGG16 to have better accuracy and EfficientNetB3 to be first place or at least compete for first place. I tuned EfficientNetB3 the most but still was not able to get better results.

To start the server go into the server directory `cd server` and start the server.py script `python server/server.py`. The server uses currently the VGG16 model with 61% accuracy.
