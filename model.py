"""Import Libraries and Packages"""
from keras.callbacks import LearningRateScheduler, CSVLogger, ModelCheckpoint
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from keras.models import load_model
from keras.optimizers import Adam, SGD
from conf import config_file as config
import matplotlib.pyplot as plt
import keras.backend as K
import PIL.Image as Image
import pandas as pd
import numpy as np
import os, re, glob, datetime, argparse, logging
import kDnCNN

#Parse arguments to the change the parameters or hyperparameters accordingly
parser = argparse.ArgumentParser()
parser.add_argument('--model', default = 'KDnCNN', type = str, help = 'Type of model')
parser.add_argument('--train_data', default = 'data/npy_data/cleanPatches.npy', type = str, help = 'Location of train data')
parser.add_argument('--batch_size', default = 128, type = int, help = 'Batch size')
parser.add_argument('--epoch', default = 1, type = int, help = 'NUmber of epoches')
parser.add_argument('--lr', default = 1e-3, type = float, help = 'Initial learning rate')
parser.add_argument('--save_every', default = 1, type = int, help = 'Save model at each epoch')
parser.add_argument('--sigma', default = 25, type = int, help = 'Level of noise')
parser.add_argument('--TestOnly', default = False, type = bool, help = 'True- Test the preTrained data and False- Train and Test')
parser.add_argument('--preTrainedModel', default = None, type = str, help = 'Location of pre trained data - DnCNN.hdf5')
parser.add_argument('--test_dir', default = 'data/Test/Set68', type = str, help = 'Location of test data')

args = parser.parse_args()

#Display the result - Images
def displayImages(x, title=None,cbar=False, size=None):
    plt.figure(figsize=size)
    plt.imshow(x, interpolation='nearest', cmap='gray')

    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    else:
        None
    plt.show()

#Save all the results
#If argument is not for test only i.e --TestOnly then
if not args.TestOnly:
    save_dir = './results/save_'+ args.model + '_' + 'sigma' + str(args.sigma) + '/'
    
    #if directory is not there then make one
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    #Add logging funtion to save the details
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%Y %H:%M:%S',
                    filename=save_dir+'info.log',
                    filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-6s: %(levelname)-6s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info(args)

else:
    save_dir = '/'.join(args.preTrainedModel.split('/')[:-1]) + '/'


#Create a function to load the  data for training
def loadData():

    trainData = np.load(args.train_data)
    logging.info("\nTrain data is loaded")
    logging.info("Size of the Train data: ({}, {}, {})".format(trainData.shape[0], trainData.shape[1], trainData.shape[2]))

    #return 
    return trainData

#Create the function for the learning rate scheduler
def learning_rate(epoch):
    init_lr = args.lr

    #if statement
    # This function keeps the learning rate at 0.001 for the first ten epochs
    # and decreases it exponentially after that.
    if (epoch < 50):
        lr = init_lr
    else:
        lr = init_lr/10

    return lr

#Create a function to train the images data
def trainData(modifiedData, batch_size=8):

    #modifiedData is used as a tesnor of patches created during
    # generating the modified data

    list_ofShape = list(range(modifiedData.shape[0]))

    #Iteration
    while(True):
        #shuffle the data
        np.random.shuffle(list_ofShape)
        for i in range(0, len(list_ofShape), batch_size):
            yBatch = modifiedData[list_ofShape[i:i+batch_size]]
            addNoise = np.random.normal(0, args.sigma/255.0, yBatch.shape)
            #Add noise in original image
            xBatch = yBatch + addNoise
            yield xBatch, yBatch

# Creae the mehtod to measure the loss function
#For this model, squared error
#Or use in-built loss function like mse, categorical_crossentropy
def custom_lossFunction(y_true, y_predict):
    dif = y_predict - y_true
    result = K.sum(K.square(dif))/2
    return result
    
#Create the function to train the model
def trainModel():

    #Call the function laodData() to train
    train_Data = loadData()
    #Give new shape
    train_Data = train_Data.reshape((train_Data.shape[0], train_Data.shape[1], train_Data.shape[2],1))
    #Specified data type
    train_Data = train_Data.astype('float32')/255.0

    #Option for the selection of the model
    if (args.preTrainedModel):
        model = load_model(args.preTrainedModel, compile=False)
    # elif (args.model == 'kDnCNN'):
    #     model = kDnCNN.kDnCNN_model()
    else :
        if args.model == 'KDnCNN': model = kDnCNN.kDnCNN_model()
    
    # Compile the model
    #model.compile(optimizer=Adam(0.001), loss=custom_lossFunction)
    #model.compile(optimizer = SGD(), loss = ['mse'])
    model.compile(optimizer = Adam(), loss = ['mse'])
    print("\nStart compiling the model")
    model.save(save_dir + '/kDnCNNmodel_{epoch:01d}.hdf5')

    #Use callbacks function API to perform actions at stages of training process
    checkpointer = ModelCheckpoint(os.path.join(save_dir,'model_{epoch:03d}.hdf5'),
                verbose=1, save_weights_only=False, period=args.save_every)
    csv_logger = CSVLogger(os.path.join(save_dir,'log.csv'), append=True, separator=',')
    lr_scheduler = LearningRateScheduler(learning_rate)

    history = model.fit_generator(trainData(train_Data,batch_size=args.batch_size),
                    steps_per_epoch=len(train_Data)//args.batch_size, epochs=args.epoch, verbose=1,
                    callbacks=[checkpointer,csv_logger,lr_scheduler])
    
    #return the model
    return model

#Create the function to test the data
def testModel(model):
    print("\nStart testing the {}".format(args.test_dir))

    #Save the test results 
    out_dir = save_dir + args.test_dir.split('/')[-1] + '/'

    #If file does not exits
    if not os.path.exists(out_dir):
        #make the directory
        os.mkdir(out_dir)

    name = []
    #Create a empty list to save data later
    #Peek signal-to-noise ratio
    psnr = []
    #Structural similarity index
    ssim = []


    #List of images in the directory
    imgList = glob.glob('{}/*.png'.format(args.test_dir))

    #Iterate through every image in the image list
    for eachImg in imgList:
        #Read the image from the directory
        originalImg = np.array(Image.open(eachImg), dtype='float32') / 255.0
        testImg = originalImg + np.random.normal(0, args.sigma/255.0, originalImg.shape)
        testImg = testImg.astype('float32')

        #Prediction
        xTest = testImg.reshape(1, testImg.shape[0], testImg.shape[1], 1)
        yPredict = model.predict(xTest)

        #Calculate the PSNR and SSIM
        newImg = yPredict.reshape(originalImg.shape)
        newImg = np.clip(newImg, 0, 1)

        psnrNoised, psnrDenoised = compare_psnr(originalImg, testImg), compare_psnr(originalImg, newImg)
        ssimNoised, ssimDenoised = compare_ssim(originalImg, testImg), compare_ssim(originalImg, newImg)

        
        # name.append('Average')
        psnr.append(psnrDenoised)
        ssim.append(ssimDenoised)
        

        #Save the images 
        #Retrive the name of the image file
        fileName = eachImg.split('/')[-1].split('.')[0]
        name.append(fileName)

        testImg = Image.fromarray((testImg*255).astype('uint8'))
        testImg.save(out_dir + fileName + '_Sigma'+'{}_PSNR{:.2f}.png'.format(args.sigma, psnrNoised))

        newImg = Image.fromarray((newImg*255).astype('uint8'))
        newImg.save(out_dir + fileName + '_PSNR{:.2f}.png'.format(psnrDenoised))
        displayImages(np.hstack((testImg,newImg)))
        

    #Add PSNRs and SSIMs to the list
    avgPSNR = sum(psnr)/len(psnr)
    avgSSIM = sum(ssim)/len(ssim)

    #Add average PSNR and SSIM to same Lists
    name.append('Average')
    psnr.append(avgPSNR)
    ssim.append(avgSSIM)

    #print
    print('\nAverage PSNR: {0:.2f}, SSIM: = {1:.2f}'.format(avgPSNR, avgSSIM))

    #save PSNRs and SSIMs to csv file
    # pd.DataFrame({'nameHeadimg':np.array(nameHeading), 'psnrList':np.array(psnrList), 'ssimList':np.array(ssimList)}).to_csv(out_dir + '/metrics.csv', index=True)
    pd.DataFrame({'name':np.asarray(name), 'psnr':np.asarray(psnr), 'ssim':np.asarray(ssim)}).to_csv(out_dir+'/metrics.csv', index=True)


if __name__ == "__main__":

    if (args.TestOnly):
        #Load the pre trained model 
        # --preTrainedModel 'location of the model'
        model = load_model(args.preTrainedModel, compile=False)
        testModel(model)
    else:
        #Call the function to train the model if above condition
        #does not meet
        model = trainModel()
        testModel(model)

