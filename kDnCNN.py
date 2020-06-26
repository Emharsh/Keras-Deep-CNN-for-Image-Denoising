"""Import Libraries and Packages"""
import numpy as np
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Input,Subtract
# from keras.callbacks import LearningRateScheduler, CSVLogger, ModelCheckpoint
# from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model

#create the function for the model
def kDnCNN_model():
    filt=64
    img_channels=1
    b_norm=True

    num_layer = 0
    input_data = Input(shape=(None,None,img_channels), name='input' + str(num_layer))

    #Starting layer
    num_layer = num_layer + 1
    #Convolution-conv2D layer with ReLu activation function for the first layer
    f_layer = Conv2D(filters=filt, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='Orthogonal', name='conv2D'+str(num_layer))(input_data)
    num_layer = num_layer + 1
    f_layer = Activation('relu', name='relu' + str(num_layer))(f_layer)

    #Increasing the depth
    #Conv2D and BatchNormalization layer with ReLu activation function for the first layer
    for i in range(15):
        num_layer = num_layer + 1
        f_layer = Conv2D(filters=filt, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='Orthogonal', use_bias=False, name='conv2D'+str(num_layer))(f_layer)
        #BatchNormalization layer
        if(b_norm):
            num_layer = num_layer + 1
            f_layer = BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'bn'+str(num_layer))(f_layer)
        num_layer = num_layer + 1
        #Activation function on a second layer
        f_layer = Activation('relu', name='relu' + str(num_layer))(f_layer)

        #Third layer i.e last
        num_layer = num_layer + 1
        f_layer = Conv2D(filters=img_channels, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='Orthogonal', use_bias=False, name='conv2D'+str(num_layer))(f_layer)
        num_layer = num_layer + 1

        #Remove the noise from the data
        f_layer = Subtract(name='subtract' + str(num_layer))([input_data, f_layer])

        model = Model(inputs=input_data, outputs=f_layer)
        #Return the model
        return model
