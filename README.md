# Dog-Breed-Classification

I found [this](https://www.kaggle.com/jessicali9530/stanford-dogs-dataset) Dataset on Kaggle on wanted to try how good I can train a Deep Learning model on it. 

My first attempt is transfer learning with a VGG16 model trained on Imagenet which only got me an accuracy of up to 61%.

Second attempt was using EfficientNetB0 and EfficientNetB3, which got me way better results than VGG16, coming in at 76%.

Best accuracy I got with NASNetLarge at 94%.

To start the server go into the server directory `cd server` and start the server.py script `python server/server.py`. The server uses currently the VGG16 model with 61% accuracy.
