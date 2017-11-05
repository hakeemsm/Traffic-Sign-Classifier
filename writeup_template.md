**Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/training_viz.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/test_img_3.jpg "Speed limit (60km/h)"
[image5]: ./examples/test_img_25.jpg "Road work"
[image6]: ./examples/test_img_29.jpg "Bicycles crossing"
[image7]: ./examples/test_img_33.jpg "Turn right ahead"
[image8]: ./examples/test_img_40.jpg "Roundabout mandatory"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. The readme.md included with the submission addresses all the rubric points. The submission includes project code

Here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

###1 Data set summary
I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 27,839
* The size of the validation set is 6960
* The size of test set is 12,630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Visualization

This is a bar chart of the training data plotted with signs on x-axis and the # on the y-axis. The plot is also included as training_viz.png in the submission

![alt text][image1]

###Design and Test a Model Architecture

####1. Data preprocessing

As a first step, the training data was shuffled

I tried grayscaling and normalizing the images but that was causing a drop in accuracy. So the network was run on the images as is


####2. Design & architecture

The design was a LeNet network with 5 layers
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Layer1:
|   Input         		| 32x32x3 RGB image   							| 
|   Convolution         | 1x1 stride, valid padding, outputs 28x28x6 	|
|   REActivation		| RELU											|
|   Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6 				    |
| Layer2:
|   Convolution 	    | 1x1 stride, valid padding, outputs 10x10x16   |
|   Activation          | RELU                                             |
|   Max pooling         | 2x2 stride, valid padding, outputs 5x5x16     |  
|   Flatten             |                                               |
| Layer3:
|   Fully connected		| Input 400 output 120        					|
|   Activation          | RELU                                          |
| Layer4:
|   Fully connected		| Input 120 output 84        					|
|   Activation          | RELU                                          |
| Layer5:
|   Fully connected		| Input 84 output 43          					|
|Softmax				| Run on test set with 5 images        									|
|						|												|
|						|												|
 


####3. Model training

To train the model, I used a LeNet architecture with 5 layers. The batch size was 128 with an epoch of 10. The weights were randomly normalized. The learning rate was set to 0.001. The AdamOptimizer was used for optimization

####4. Validation approach

Various approaches were tried for improving the accuracy such as grayscaling and normaling the images, adjusting the batch size, and tuning the learning rate. Grayscaling and normalzing were causing the accuracy to drop drastically therefore those techniques were not applied for any of the datasets. Increasing the batch size to 20 didn't have any noticeable difference in performance and as a result it was kept at 10. Changing the learning rate to 0.01 also was causing a 40% reduction in accuracy and because of that it was maintained at 0.001

My final model results were:
* training set accuracy of 97.7%
* validation set accuracy of 94.9%
* test set accuracy of 85%
* accuracy on 5 random images found on the internet 20%


If a well known architecture was chosen:
* What architecture was chosen?
* The LeNet5 architecture was chosen because of its capability in handling a variety of image types 
* Why did you believe it would be relevant to the traffic sign application?
* LeNet5 performed well on the MNIST dataset which comprises of vastly varying image patterns. Compared to that, the German traffic sign dataset has well defined images and not as much variety as MNIST. So it was a logical choice
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
* The model had high accuracy on training and validation data and even on the test data with only randomly chosen images, it performed pretty consistently
 

###Test a Model on New Images

####1. Five images for German traffic signs were downloaded from Google image gallery



![Speed limit (60km/h)][image4] ![Road work][image5] ![Turn right ahead][image6] 
![Bicycles crossing][image7] ![Roundabout mandatory][image8]

The road work and bicycles images would have been diffict to classify since they have complicated patterns within the image triangles

####2. Model's prediction on the 5 random images from the internet

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (60km/h)	| Speed limit (50km/h)							| 
| Road work    			| Right-of-way at the next intersection			|
| Turn right ahead		| Ahead only									|
| Bicycles crossing		| Slippery road                         		|
| Roundabout mandatory	| Roundabout mandatory 							|


The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%. This is not as good as the test accuracy from the images provided in the test.p files since the images were from the internet were of different sizes and format and had to be resized 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 24th cell and the values are printed out in the 27th cell of the Ipython notebook.

For the first image, the model was reasonably sure that this it was a 60 km/h sign (probability of 0.6), since the prediction it made was for a 50km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.6          			| Speed limit (60km/h)							| 
| 0     				| ROad work 									|
| 0.8					| Turn right ahead								|
| 0	      			    | Bicycles crossing				 				|
| 1				        | Roundabout mandatory 							|


For the second image, the model was not even close as it predicted Right of way for the Road work sign
For the third image, the model was almost accurate in its prediction with a probablity of 0.8
For the fourth image again it didn't even come close since the image didn't even make it to the top 5 predictions
The fifth image was predicted accurately with a probability of 1 for the roundabout sign

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


