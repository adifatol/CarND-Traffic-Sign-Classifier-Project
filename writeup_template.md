# **Traffic Sign Recognition** 

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

[image1]: ./writeup_imgs/class-distribution.png "Class Dist Visualization"
[image2]: ./writeup_imgs/b_gray.png "Before Grayscaling"
[image3]: ./writeup_imgs/a_gray.png "After Grayscaling"
[image4]: ./writeup_imgs/t_orig.png "Transformed Sign Original"
[image5]: ./writeup_imgs/t_1.png "Transformed Sign 1"
[image6]: ./writeup_imgs/t_2.png "Transformed Sign 2"
[image7]: ./writeup_imgs/t_3.png "Transformed Sign 3"
[image8]: ./writeup_imgs/lenet5.png "Lenet5 architecture"
[image9]: ./from-google/p1.jpg "From google image 1"
[image10]: ./from-google/p2.jpg "From google image 2"
[image11]: ./from-google/p3.jpg "From google image 3"
[image12]: ./from-google/p4.jpg "From google image 4"
[image13]: ./from-google/p5.jpg "From google image 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

 - This file is the writeup that includes the required rubric points.
 - Here is a link to my [project code](https://github.com/adifatol/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### Summary statistics
I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32, 32, 3
* The number of unique classes/labels in the data set is 43

#### Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of training / traffic signs data over the different classes. It is noticeable that some traffic signs have much more training examples than others.

![alt text][image1]

### Design and Test a Model Architecture

#### Grayscale
First step was to transform all images to grayscale. This benefits greatly to the performance in the next processing steps and also in the training itself. Also it might improve the accuracy during the training because we get loose some extra information (color / light) which may act as useless noise.

![alt text][image2]   ![alt text][image3]

#### Normalization
This step ensures that all the data is on the same scale and can have a couple of benefits, such as improving the math (not getting large numbers for parameters) and improving the performance during the training. The images look the same as the data contains the same information, only on a different scale.

#### Data augmentation
Additional data is very useful at least for classes that don't have enough examples in the training set. So, for every class that had less than 700 examples, it was generated another 3 samples for each existing image. This means that for the lowest represented classes (eg. Class 0) with 180 examples, in the end there are 720 examples. A total of 27357 extra images were added to the training examples set.
The method to generate the extra data is to randomly apply the folowing operations:
    - rotate left/right randomly from -5 to +5 degrees.
    - scale up by adding from 2 to 6 extra width and cropping the image (so we have the same size of 32x32)
    - move the image by random +/- 2 pixels to left/right and up/down

Here is an example of an original image and an augmented image(s):

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7]

The extra images should help reduce the overfitting during the training and get a better accuracy on the validation & test sets.

#### Model Architecture

The 'LeNet' model was used in the project, having the structure as in the next table:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32 Grayscale images, with 1 color channel. The images are "squeezed" so instead of 32x32x1, they have the shape 32x32. | 
| Convolution     	| 5x5 stride, 6 feature maps, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, 6 feature maps, outputs 14x14x6 				|
| Convolution	    | 5x5 stride, valid padding, outputs 10x10x16      									|
| Max pooling	      	| 2x2 stride, 16 feature maps, outputs 5x5x6 				|
| Flatten		| flattens the matrix to a single dim array, outputs 400									|
| Fully Connected				| input 400, output 120        									|
| RELU					|												|
| Fully Connected				| input 120, output 84        									|
| RELU					|												|
| Fully Connected				| input 84, output 43									|

#### Training
The model was trained using the AdamOptimizer (implementing the [Adam gradient descent](https://arxiv.org/abs/1412.6980) algorithm)
The batch size used for training was 140 images.
The optimization was repeated over 70 epochs at a training rate of 0.002.
Other parameters used:
    mu = 0
    sigma = 0.1
    keep_prob = 0.55 (for dropout, not present in the final solution)

#### Solution
The solution used is LeNet implementation from the [CarND LeNet Lab](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb).
The implementation initially was performing at ~0.89 acuracy on test and validation sets.
The training rate changed from 0.0001, 0.0003 ... to 0.005 but the current one gives good results (fast enough conversion) for the project requirements. Adding a dropout layer did not help much, even with various keep probabilities from 0.2 up to 0.8.
The data augmentation improved alot the results, given enough epochs. This was the most useful improvement.

The final model results are:
 * Test Accuracy = 0.921
 * Validation Accuracy = 0.951

Probably adding a convolutional layer and a "correctly placed" dropout layer would help improving the test accuracy, as the current solution seems to overfit.

### Test a Model on New Images

#### Chosen images

Here are five German traffic signs found on the web:

![alt text][image9] ![alt text][image10] ![alt text][image11] 
![alt text][image12] ![alt text][image13]

The first image might be difficult to classify probably because the number of training examples for this class is not that big.
Second image should be quite good represented in the training set.
Third image also should be predicted corectly.
The forth image is not good represented but also it looks a bit "warped" so probably it will be difficult to recognize it.
The fifth image is actually not present in the training set, but looks very similar to the image "No passing for vehicles over 3.5 metric tons"

#### Prediction

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:| 
| Turn right ahead      		| Stop   									              | 
| Speed limit (70km/h)    | Speed limit (70km/h) 									|
| Speed limit (60km/h)				| Keep left          											|
| Stop           	      		| Stop                					 				|
| No passing for vehicles over 3.5 metric tons			| No passing  |

The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%.

#### Top 5 Softmax

The softmax probabilities are calculated in the last but one cell in the jupyter file. The results seem a little wierd, as they are not in the 0 - 1 range.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 13.60344315         			| Turn right ahead   									| 
| 154.18362427     				| Speed limit (70km/h) 										|
| 16.57634163					| Speed limit (60km/h)											|
| 53.98876572	      			| Stop					 				|
| 23.30830002				    | No passing for vehicles over 3.5 metric tons      							|

