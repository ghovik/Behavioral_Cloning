# Behavioral Cloning Project

### Goals / steps:

* Use the simulator to manually drive the vehicle to collect data of good driving behavior
* Build a convolutional neural network in Keras that predicts steering angles from data
* Train and validate the model with a training and validation set
* After training the model, test it in the simulator in autonomous mode
* The general goal is to let the vehicle learn and drive smoothly, the least requirement is that the vehicle successfully drive one track without leaving the road.



### Model Architecture and Training Strategy

Two model architectures - LeNet and Nvidia - have been tested and finally Nvidia architecture was selected as our solution. The data is normalized and mean centralized in the model using a Keras lambda layer.

The model contains max-pooling layers in order to reduce overfitting. The model used an 'adam' optimizer, so the learning rate was not tuned manually. The model was trained and validated on different data sets to ensure that the model was not overfitting. Different datasets, including clock-wise loops, counter clock-wise loops, recovering, smoothly driving on sharp turns are recorded for training. The model was tested on different combinations of them. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

`video.mp4` shows the final resulting driving behavioral by the vehicle itself after learning.