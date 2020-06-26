#Import OS python library
import os

#Batch size for the training
batch_size = 128

#Number of epochs
epoch = 1

#Level of noise for the training
sigma = 20.0

#Location to save the patches
save_dir = './data/npy_data/'

# Training data path
# data = './training/image_clean.npy'

#Location where generated data is saved 
# genDataPath = './genData/'
src_dir = './data/Train400/'

#To generate patch - patch size, stride and step required
pat_size, stride = 40, 10
step = 0
#stride = 10

#Number of Augmentation time
augmentTimes = 1

iterat = 2000
num_epoch = 5
