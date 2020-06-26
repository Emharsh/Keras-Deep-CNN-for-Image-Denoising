# import the libraries
import os
import numpy as np
from cv2 import cv2
import glob
from conf import config_file as config
from pathlib import Path
from multiprocessing import Pool

# define scales which will be used to rescale the images for
#generating the patches
scales=[1,0.9,0.8,0.7]

#Create the function for the data augmentation
def dataAug(img, mode = 0):
    if(mode == 0):
        return img
    elif(mode == 1):
        return np.flipud(img)
    elif(mode == 2):
        return np.rot90(img)
    elif(mode == 3):
        return np.flipud(np.rot90(img))
    elif(mode == 4):
        return np.rot90(img, k=2)
    elif(mode == 5):
        return np.flipud(np.rot90(img, k=2))
    elif(mode == 6):
        return np.rot90(img, k=3)
    elif(mode == 7):
        return np.flipud(np.rot90(img,k=3))

#Create the function to generate patches
def patches(filename):
    #Read the image from the file
    img = cv2.imread(filename, 0)
    img_h, img_w = img.shape
    # img_w = img.shape
    patches = []

    #Scalling of the images
    for s in scales:
        imgH_scaled,imgW_scaled = int(img_h*s),int(img_w*s)
        # imgW_scaled = int(img_h*s),int(img_w*s)
        img_s = cv2.resize(img, (imgH_scaled, imgW_scaled), interpolation=cv2.INTER_CUBIC)
        #Extracting the patches of the images
        for x in range(0+config.step,(imgH_scaled-config.pat_size+1), config.stride):
            for y in range(0+config.step,(imgW_scaled-config.pat_size+1), config.stride):
                ext = img_s[x:x+config.pat_size, y:y+config.pat_size]
                for k in range(0, config.augmentTimes):
                    img_aug = dataAug(ext, mode=np.random.randint(0,8))
                    #Add patches
                    patches.append(img_aug)
    # print("Generated Patches\n", patches)
    # print("Generated patches: ", len(patches))
    return patches


if __name__ == '__main__':
    # data = genData(data_dir = 'data/npy_data')

    #Retrieve the modified version of image from source directory
    imgList = glob.glob(config.src_dir + '*.png')
    #Divide it into more running tasks
    nThreads = 16

    #Create a list to save modified images
    data = []

    #Generates patches by using patches() function
    for i in range(0, len(imgList), nThreads):
        #Use pool to map the input to the different processors
        #to increase the speed
        multiProcess = Pool(nThreads)
        pat = multiProcess.map(patches, imgList[i:min(i+nThreads, len(imgList))])

        print("\nStart adding pacthes to the images\n")
        #Add patches
        for p in pat:
            data = data + p


    # print('Images '+str(i)+'-'+str(i+nThreads))
    print('Patches are added\n')

    #Now save the modified data in .npy format
    data = np.array(data, dtype = 'uint8')
    # print("\nTotal patches: ", str(data))
    print("Batch size: ", config.batch_size)
    print("Modified shape: ", str(data.shape))
    print("Data is saved in cleanPatches.npy")

    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)
    np.save(config.save_dir+'cleanPatches.npy', data)
