# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./results/dataset.png "Dataset"
[image2]: ./results/structure.png "NN"
[image4]: ./german_traffic_signs/sing_1.jpg "Traffic Sign 1"
[image5]: ./german_traffic_sings/sign_2.jpg "Traffic Sign 2"
[image6]: ./german_traffic_sings/sign_3.jpg "Traffic Sign 3"
[image7]: ./german_traffic_sings/sign_4.jpg "Traffic Sign 4"
[image8]: ./german_traffic_sings/sign_5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used python and numpy methods to calculate the stadistics of the provided traffic signs dataset. The obtained characteristics are:

* The size of the training set is 34799.

* The size of the validation set is 4410.

* The size of the test set is 34799.

* The shape of a traffic sign image is 32x32x3.

* The are 43 unique classes/labels in this data set.

#### 2. Include an exploratory visualization of the dataset.

Firstly, I decided to visualize 10 images of the training set. 

![alt text][image1]

Then, to get a wider perspective of the content in the dataset, I plotted, for each its partitions, how many images per class there are. 

<img src=./results/train_set.png width="400">
<img src=./results/validation_set.png width="400">
<img src=./results/test_set.png width="400">

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Although pre-processing is encourage during the development of this project, I focused more on the design and re-modeling of the neural network in order to obtained the demanded accuracy rather than apply multiple pre-processing techniques. Therefore, I only normalized the datasets calling `normalize()` function. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model is based in the LeNet architecture explained and designed during the class. However, I have included two dropout layers between the two fully connected layers with a 90% ratio. Moreover, I have modified the number of filters used in the middle convolutional layers.

The structure of the neural network is: 

![alt text][image2]

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Eventually, the characteristics of the trained model are: 

* Optimizer: Adam 

* Batch Size: 250

* Epochs: 30 

* Learning rate: 0.001

* Mean: 0 

* Variance: 0.1

* Dropout layers: 0.9 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

To begin, I iterated the number of epochs value in order to see how much the network needed to reach the desired value. Concluding that actually not many epochs were needed in order to achieve the desired accuracy value, I decided to set the training to 10 epochs. However, modifications afterwards the results of the network were not certainly good, which led me to increase the number of epochs to 30 and the batch size from 128 to 250. 
Then, I decided to play around different values for the introduced dropout layers and how many filters would be applied in the middle layers. Further, I decided to modify the values of the fully connected layers. 
Finally, I have obtained 0.937 accuracy for the validation set, 0.999 for the training set and 0.987 for the test set. 

Results per epoch: 
```
EPOCH 1 ...
Validation >> Accuracy = 0.780 Loss = 0.788
Train >> Accuracy = 0.858 Loss = 0.509

EPOCH 2 ...
Validation >> Accuracy = 0.854 Loss = 0.526
Train >> Accuracy = 0.939 Loss = 0.206

EPOCH 3 ...
Validation >> Accuracy = 0.888 Loss = 0.432
Train >> Accuracy = 0.966 Loss = 0.126

EPOCH 4 ...
Validation >> Accuracy = 0.905 Loss = 0.392
Train >> Accuracy = 0.981 Loss = 0.078

EPOCH 5 ...
Validation >> Accuracy = 0.902 Loss = 0.398
Train >> Accuracy = 0.982 Loss = 0.065

EPOCH 6 ...
Validation >> Accuracy = 0.905 Loss = 0.392
Train >> Accuracy = 0.989 Loss = 0.044

EPOCH 7 ...
Validation >> Accuracy = 0.902 Loss = 0.404
Train >> Accuracy = 0.991 Loss = 0.034

EPOCH 8 ...
Validation >> Accuracy = 0.928 Loss = 0.360
Train >> Accuracy = 0.995 Loss = 0.020

EPOCH 9 ...
Validation >> Accuracy = 0.924 Loss = 0.304
Train >> Accuracy = 0.995 Loss = 0.019

EPOCH 10 ...
Validation >> Accuracy = 0.924 Loss = 0.330
Train >> Accuracy = 0.993 Loss = 0.025

EPOCH 11 ...
Validation >> Accuracy = 0.913 Loss = 0.400
Train >> Accuracy = 0.995 Loss = 0.018

EPOCH 12 ...
Validation >> Accuracy = 0.926 Loss = 0.406
Train >> Accuracy = 0.997 Loss = 0.012

EPOCH 13 ...
Validation >> Accuracy = 0.923 Loss = 0.402
Train >> Accuracy = 0.997 Loss = 0.013

EPOCH 14 ...
Validation >> Accuracy = 0.919 Loss = 0.386
Train >> Accuracy = 0.998 Loss = 0.010

EPOCH 15 ...
Validation >> Accuracy = 0.935 Loss = 0.315
Train >> Accuracy = 0.997 Loss = 0.014

EPOCH 16 ...
Validation >> Accuracy = 0.930 Loss = 0.331
Train >> Accuracy = 0.998 Loss = 0.010

EPOCH 17 ...
Validation >> Accuracy = 0.919 Loss = 0.473
Train >> Accuracy = 0.996 Loss = 0.015

EPOCH 18 ...
Validation >> Accuracy = 0.932 Loss = 0.365
Train >> Accuracy = 0.998 Loss = 0.009

EPOCH 19 ...
Validation >> Accuracy = 0.931 Loss = 0.418
Train >> Accuracy = 0.998 Loss = 0.009

EPOCH 20 ...
Validation >> Accuracy = 0.927 Loss = 0.361
Train >> Accuracy = 0.996 Loss = 0.014

EPOCH 21 ...
Validation >> Accuracy = 0.922 Loss = 0.465
Train >> Accuracy = 0.997 Loss = 0.010

EPOCH 22 ...
Validation >> Accuracy = 0.925 Loss = 0.502
Train >> Accuracy = 0.999 Loss = 0.005

EPOCH 23 ...
Validation >> Accuracy = 0.934 Loss = 0.351
Train >> Accuracy = 0.999 Loss = 0.005

EPOCH 24 ...
Validation >> Accuracy = 0.931 Loss = 0.402
Train >> Accuracy = 0.997 Loss = 0.009

EPOCH 25 ...
Validation >> Accuracy = 0.932 Loss = 0.396
Train >> Accuracy = 0.998 Loss = 0.007

EPOCH 26 ...
Validation >> Accuracy = 0.934 Loss = 0.369
Train >> Accuracy = 0.998 Loss = 0.007

EPOCH 27 ...
Validation >> Accuracy = 0.932 Loss = 0.394
Train >> Accuracy = 0.998 Loss = 0.005

EPOCH 28 ...
Validation >> Accuracy = 0.936 Loss = 0.332
Train >> Accuracy = 0.997 Loss = 0.013

EPOCH 29 ...
Validation >> Accuracy = 0.928 Loss = 0.339
Train >> Accuracy = 0.999 Loss = 0.009

EPOCH 30 ...
Validation >> Accuracy = 0.937 Loss = 0.360
Train >> Accuracy = 0.999 Loss = 0.003
```

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

<img src=./german_traffic_signs/sing_1.jpg width="200"> <img src=./german_traffic_signs/sing_2.jpg width="200"> <img src=./german_traffic_signs/sing_3.jpg width="200"> <img src=./german_traffic_signs/sing_4.jpg width="200"> <img src=./german_traffic_signs/sing_5.jpg width="200">

I believe that these selected images may be difficult to classify as they have not been adjusted to display only the part of the image that contains the signal information, meaning that the network might be confused by their background, for example. 


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 60 km/h       		| Traffic signals   							| 
| General Caution		| Roundabout mandatory 							|
| Yield					| Yield											|
| Ahead only	      	| Children crossing				 				|
| Road work 			| Right-of-way at the next intersection			|


The model was able to correctly guess only 1 of the 5 traffic signs, which gives an accuracy of 20%. This result is quite poor, proving the the network is not strong enough to work with images from outside the officials dataset. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 18th cell of the Ipython notebook. The results of the predictions can be observed in the following image: 

<img src=./results/softmax.jpg width="400">


