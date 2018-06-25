# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./examples/summary_stats.png "Visualization"
[image2]: ./web-tests/german_sign.jpg "Traffic Sign 1"
[image3]: ./web-tests/german_sign2.jpg "Traffic Sign 2"
[image4]: ./web-tests/german_sign3.jpg "Traffic Sign 3"
[image5]: ./web-tests/german_sign4.jpg "Traffic Sign 4"
[image6]: ./web-tests/german_sign5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ignaden/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

* The size of training set was originally 34,799 obveservations. As I describe below, I will remove some of the points for categories with excessive representation (I chose 1,100 as the maximum per group).

* The size of the validation set is 4,410 observations.

* The size of test set is 12,630 observations.

* The shape of a traffic sign image is 32x32 pixels. The original format is RGB, so 3 channels (32x32x3), but I will convert it to grayscale, hence it'll be 32x32x1.

* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here's a distribution of the number of labels represented in each category within the training set. I will 'trim' each category

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

At first attempt, I made no changes to the proposed architecture. I could not get more than 90% accurace of the validation data.

My next step was to convert images to grascale. Here's the code that does this (it's all in the preprocessing cell):

```python
def rgb_to_gray(X):
    images_gray = np.average(X, axis=3)
    images_gray = np.expand_dims(images_gray, axis=3)
    return images_gray

def preprocess_data(X, y, norm=False, mu=None, stddev=None):
    # Convert from RGB to grayscale if applicable
    if GRAYSCALE:
        X = rgb_to_gray(X)
    #elif YUVSPACE:
    #    X = rgb_to_yuv(X)   

    if norm:
        mu = np.mean(X)
        stddev = np.std(X)
        
        return (X - mu) / stddev, y, mu, stddev
    else:
        return (X - mu) / stddev, y
```

The results became marginally better, but I still could not get 93%. 

My next step was to normalise the data. The code is included in the snippet above.
When constructing the actual data, I had to calcualte the mean and standard deviation for the training data and use those numbers to also scale the validation and testing data as well. It would be wrong to scale those  data sets using sample mean and standard deviations of those data sets.

```python
# training
X_train_p, y_train_p, n_mu, n_stddev = preprocess_data(X_train, y_train, True)
X_train_p, y_train_p, tot = remove_excess(X_train_p, y_train_p, 1100)

print ('Total removed: %.1d. Remaining to train with %.1d' % (tot, X_train_p.shape[0]))

# validation
X_valid_p, y_valid_p = preprocess_data(X_valid, y_valid, False, n_mu, n_stddev)

# testing
X_test_p, y_test_p   = preprocess_data(X_test, y_test, False, n_mu, n_stddev)
```

The results did not improve significantly, and were still below the 93% threshold. My next step was to take out 
excessively represented labels (you'll see in the code above, I used threshold of 1,100 as the maximum number of
points in the dataset for each label). This will 

Here's the code that did the removal of excess data:
```python
def remove_excess (x_d, y_d, max_num=700):
    total_removed = 0
    for i in range (43):        
        # Get the indices
        curr_labels = y_d == i

        # How many do we have from this label?
        l = np.sum(curr_labels)
        
        if l > max_num:
            l = np.sum(curr_labels)

            indices = np.where(curr_labels)
            extra = indices[0][(max_num-1):]

            x_d = np.delete(x_d, extra, 0)
            y_d = np.delete(y_d, extra, 0)
            
            total_removed += extra.size
            print ('For idx = %.1d; we have %.1d. deleted %.1d' % (i, l, extra.size))
    
    return x_d, y_d, total_removed
```

At this point, the validation accuracy went above 93%, so I stopped the improvements. The next step would've been
to artifically create more data so that each label is represented with more or less equivalent number of points.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image                       | 
| Convolution 3x3     	| 5x5 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 3x3	    | 1x1 stride,  outputs 10x10x16                 |
| RELU                  |                                               | 
| Max pooling	      	| 1x1 stride,  outputs 5x5x16				    |
| Flatten               | Outputs 400                                   |
| Fully connected		| Input = 400. Output = 120                     |
| RELU                  |                                               |
| Fully connected		| Input = 84. Output = 43                       |
| Softmax				| 43        									| 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I followed the model suggested within the course materials using `AdamOptimizer`. Here're the final configuration:

```python
rate = 0.001

EPOCHS = 50
BATCH_SIZE = 128

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
```

I used the learning rate hyperparameter as was suggested and increased the number of Epochs to 50, as it seemed that it converged by the then.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 93.5%
* test set accuracy of 92.2%

I used the model suggested by the course materials, hence I will answer the following:

If a well known architecture was chosen:
* What architecture was chosen? LeNet
* Why did you believe it would be relevant to the traffic sign application? The course suggested it and this is a famous model with well-documented results for this type of tasks.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? In all three categories, the results are very good - over 90%. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web (note that I had to format them to 32x32):

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]

The model makes a sigle error with `Stop Sign` - it's mostly likely because of the poor image quality.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                        |     Prediction	        					| 
|:-------------------------------------:|:---------------------------------------------:| 
| General caution                       | General caution                               | 
| Speed limit (30km/h)                  | Speed limit (30km/h)                          |
| Right-of-way at the next intersection | Right-of-way at the next intersection         |
| Keep left	      		                | Keep left					 				    |
| Stop			                        | Vehicles over 3.5 metric tons prohibited      |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is slightly worse than the resulting accuracy of the actual test data, which is 92%. Of course, the sample size is small, hence the deviation.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the cell following `Predict the Sign Type for Each Image`.

For all of the predictions, the model gave 100% for each of the predicted labels. I presume this may be the result of overfitting, but difficult to tell without further investigation.

| Probability   |     Prediction	        					| 
|:-------------:|:---------------------------------------------:| 
| 100%          | General caution                               | 
| 100%          | Speed limit (30km/h)                          |
| 100%          | Right-of-way at the next intersection         |
| 100%          | Keep left					 				    |
| 100%          | Vehicles over 3.5 metric tons prohibited      |

