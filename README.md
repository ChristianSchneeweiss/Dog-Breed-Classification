# Dog-Breed-Classification

I found [this](https://www.kaggle.com/jessicali9530/stanford-dogs-dataset) Dataset on Kaggle on wanted to try how good I can train a Deep Learning model on it. The Kaggle Kernel I created - [kernel](https://www.kaggle.com/waterchiller/vgg16-classification-dog-breed)

1. NASNetLarge 94%
2. InceptionResNetV2 83%
3. NASNetMobile 81%
4. InceptionV3 77%
5. EfficientNetB3 76%
6. MobileNetV2 73% (alpha=1.0)
7. EfficientNetB0 68%
8. VGG16 61%

I expected VGG16 to have better accuracy and EfficientNetB3 to be first place or at least compete for first place. I tuned EfficientNetB3 the most but still was not able to get better results.

To start the server go into the server directory `cd server` and start the server.py script `python server/server.py`. The server uses currently the VGG16 model with 61% accuracy.
