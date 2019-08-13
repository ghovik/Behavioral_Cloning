import csv
import cv2
import numpy as np
import sklearn
import random
from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

rate = 0.0001
Epochs = 3
cor = 0.2
# read csv, put in lines
samples = []
path_img = 'nvplus/IMG/'
driving_log = 'nvplus/driving_log.csv'
with open(driving_log) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name_center = path_img+batch_sample[0].split('/')[-1]
                #name_left = path_img+batch_sample[1].split('/')[-1]
                #name_right = path_img+batch_sample[2].split('/')[-1]
                center_angle = float(batch_sample[3])
                center_image = cv2.imread(name_center)

                images.append(center_image)
                angles.append(center_angle)
                #images.append(cv2.flip(center_image,1))
                #angles.append(-center_angle)

                #left_image = cv2.imread(name_left)
                #images.append(left_image)
                #angles.append(center_angle+cor)
                #images.append(cv2.flip(left_image,1))
                #angles.append(-center_angle-cor)

                #right_image = cv2.imread(name_right)
                #images.append(right_image)
                #angles.append(center_angle-cor)
                #images.append(cv2.flip(right_image,1))
                #angles.append(-center_angle+cor)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Nvidia model
def nvidia_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((60,20),(0,0))))
    model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2,2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2,2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2,2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2,2), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2,2), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    #model.compile(loss='mse', optimizer=Adam(rate))
    model.compile(loss='mse', optimizer='adam')

    return model

model = nvidia_model()

history_obj = model.fit_generator(train_generator,
            samples_per_epoch=len(train_samples),
            validation_data=validation_generator,
            nb_val_samples=len(validation_samples),
            nb_epoch=Epochs)

model.save('cliu.h5')
