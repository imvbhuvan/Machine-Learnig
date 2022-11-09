import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Conv2D,MaxPool2D
from tensorflow.keras.models import Sequential


#Rescale the dataset using the imagedatagenerator
train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)
#import the train data with target_size and bactch_Size
train_dataset=train.flow_from_directory(r"path for train data",target_size=(150,150),batch_size=32,class_mode='binary')
#import the test datafrom with parametes
test_dataset=test.flow_from_directory(r"path for test data",target_size=(150,150),batch_size=32,class_mode='binary')

#Check how many class are present
test_dataset.class_indices

#Install the Sequential model
model = Sequential()
#Add the conv2d layer to the model with relu activation function and input sahp
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
#add the another maxpoolind layer to the model
model.add(MaxPool2D(2,2))
#add another convluation layer to the model
model.add(Conv2D(64,(3,3),activation='relu'))
#Add the another max pooling layer
model.add(MaxPool2D(2,2))
# Add another Covlution layer with relu activation function
model.add(Conv2D(128,(3,3),activation='relu'))
#Add the another max pooling layer
model.add(MaxPool2D(2,2))
#Add another Covlution layer with relu activation function
model.add(Conv2D(128,(3,3),activation='relu'))
#Add the another max pooling layer
model.add(MaxPool2D(2,2))
#Add the flattern layer to the model
model.add(Flatten())
#Add the dense layer to the model with relu activation function
model.add(Dense(512,activation='relu'))
#Add the dense layer to the model with sigmoid activation function
model.add(Dense(1,activation='sigmoid'))

#compile the model with adam optimizet and loss function we use the accu
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fitting the defined model
r = model.fit(train_dataset,epochs = 10,validation_data = test_dataset)

# plotting the loss
plt.plot(r.history['loss'], label='train loss')
#plot the val_loss function
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# plotting the accuracy
plt.plot(r.history['accuracy'], label='train acc')
#plot the val_accuracy score
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

#Creating a function the model is predict the input images with good accuracy
def predictImage(filename):
#create a varibale and load_image with target_size
        img1 = image.load_img(filename,target_size=(150,150))
#Let's visualize it using the matplotlib
        plt.imshow(img1)
#Create y variable to covert the image into arrys
        Y = image.img_to_array(img1)
#Expand the shape of an array.
#Insert a new axis that will appear at the axis position in the expanded a
        X = np.expand_dims(Y,axis=0)
#Predict the the test dataset
        val = model.predict(X)
        print(val)
#create condition to for predict the labels
        if val == 1:
            plt.xlabel("No Fire",fontsize=30)
        elif val == 0:
            plt.xlabel("Fire",fontsize=30)
            
#get predictions on certain images
predictImage(r'image path to ger predictions')
