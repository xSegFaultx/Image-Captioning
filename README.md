# Image-Captioning
[![Udacity Computer Vision Nanodegree](https://img.shields.io/badge/Udacity-Computer%20Vision%20ND-deepskyblue?style=flat&logo=udacity)](https://www.udacity.com/course/computer-vision-nanodegree--nd891)
[![Pytorch](https://img.shields.io/badge/%20-Pytorch-grey?style=flat&logo=pytorch)](https://pytorch.org/) \
In this project, I designed a CNN-RNN model that could automatically generate captions for images. In this model, 
the CNN (Convolutional Neural Network) acts like an encoder that extracts features from the input image and 
the RNN (Recurrent neural network) acts like a decoder that decodes the features and generates 
captions for the input image. 
The CNN - RNN model design follows the model structure described in ["Show and tell: A neural image caption generator. 2015"](https://arxiv.org/pdf/1411.4555.pdf) by Vinyals, O., Toshev, A., Bengio, S., and Erhan, D. The general structure of the model is shown in Fig 1.0. 
The model is trained on the [COCO dataset](https://cocodataset.org/) by Microsoft. 
<figure>
<img src="https://github.com/xSegFaultx/Image-Captioning/raw/master/images/encoder-decoder.png" alt="encoder-decoder">
<figcaption align = "center"><b>Fig.1.0 - Structure of the CNN-RNN model </b></figcaption>
</figure>

# CNN - RNN Model
## CNN Encoder
For the CNN encoder, I used a pre-trained ResNet-50. 
Since I am only using ResNet as a feature extractor instead of an image classifier, I removed the fully connected layers (classifier) from ResNet. 
The extracted features are flattened to a feature vector and then passed to a linear layer. 
The structure of the encoder is shown in Fig 2.0. 
The linear layer transforms and resizes the feature vector to match the size of the word embedding (input size of the RNN). 
Since the pre-trained ResNet already did a great job extracting features from images, I decided not to re-train the network. 
Therefore, from the encoder, only the linear layer is trained during the training phase.
<figure>
<img src="https://github.com/xSegFaultx/Image-Captioning/raw/master/images/encoder.png" alt="CNN encoder">
<figcaption align = "center"><b>Fig.2.0 - Structure of the CNN encoder </b></figcaption>
</figure>

## RNN Decoder
For the RNN decoder, I used an RNN with LSTM cell. 
For each image, the ground truth caption of the image is first mapped to the embedding space. 
The embedded image feature vector from the encoder is also mapped to the embedding space to match the input of the RNN. 
The input to the decoder is formed by concatenating the feature vector with ground truth caption as shown in Fig 3.0. 
The input is passed to an LSTM cell followed by 3 linear layers. 
Since the output caption sequence could be as long as 20 tokens, I used an LSTM cell instead of the vanilla RNN. 
Also, since the feature vector is the first "token" passed to the decoder, the token near the end of the caption may have little to no dependency on the feature vector if I used a vanilla RNN model. 
<figure>
<img src="https://github.com/xSegFaultx/Image-Captioning/raw/master/images/decoder.png" alt="RNN decoder">
<figcaption align = "center"><b>Fig.3.0 - Structure of the RNN decoder </b></figcaption>
</figure>
The LSTM cell "combines" information from the current word embedding (or image feature vector) and "memory" from the previous LSTM cells and finally passes this processed information to the linear layers. 
The linear layers will predict a score for each token in the vocabulary based on the information from the LSTM cell. 
The token with the highest score is the output of the decoder. 
The state of the current LSTM and the predicted token (or the ground truth token if the network is in the training phase) is passed to the next LSTM cell to generate the next token. 
In this way, we can use the feature vector from the encoder to generate a complete caption for the image.

# Result
After training for 3 epochs, the perplexity of the model has reduced to around 7.0 which is pretty good for our model even though we are using a very small vocabulary (around 8000). 
I sampled a few images from the testing set and used my trained model to generate captions for these images. The results are shown in Fig 4.0.
<figure>
<img src="https://github.com/xSegFaultx/Image-Captioning/raw/master/images/result1.PNG" alt="test result 1">
<img src="https://github.com/xSegFaultx/Image-Captioning/raw/master/images/result2.PNG" alt="test result 2">
<figcaption align = "center"><b>Fig.4.0 - Testing Result </b></figcaption>
</figure>
However, the model is not perfect. As shown in Fig 4.1, the model could misclassify certain objects in an image or completely misunderstand the entire image. 
<figure>
<img src="https://github.com/xSegFaultx/Image-Captioning/raw/master/images/result3.PNG" alt="incorrect test result 1">
<img src="https://github.com/xSegFaultx/Image-Captioning/raw/master/images/result4.PNG" alt="incorrect test result 2">
<figcaption align = "center"><b>Fig.4.1 - Incorrect Testing Result </b></figcaption>
</figure>
From Fig 4.1 we can see that the sentence (caption) generated by the model is grammatically correct and also makes sense to human readers. 
Therefore, I think that there might be limitations on the encoder side of the model instead of the decoder side. 
Since I did not re-train the ResNet model on the COCO dataset, the accuracy of the encoder might not be that great. 
For example, the features extracted from an image of a hotdog and an image of a piece of bread with a human thumb pressed on it may be quite similar. 
Also, I could train the decoder for more epochs so that it can distinguish nuances between similar feature vectors.

# WARNING
This is a project from Udacity's ["Computer Vision Nanodegree"](https://www.udacity.com/course/computer-vision-nanodegree--nd891). I hope my code can help you with your project (if you are working on the same project as this one) but please do not copy my code and please follow [Udacity Honor Code](https://www.udacity.com/legal/community-guidelines) when you are doing your project.
