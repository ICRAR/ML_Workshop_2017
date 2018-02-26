
##Imports:

import os
import numpy as np
import glob
import cv2
from datetime import datetime
import random
import string
import time
import sys
import scipy
import csv
import tensorflow as tf


#Start timing
start = datetime.now()
print "********************************************"

##Defining some stuff:

img_dir = "data/images/training/train/"
base_dir = "/home/keegansmith/keras/ML/"
#base_dir = ""
#train_dir = "train/"
#valid_dir = "valid/"

learning_rate = 1e-6
loss_fn = "categorical_crossentropy"

if len(sys.argv)==5:
    train_samples = int(sys.argv[1])
    test_samples =int(sys.argv[2])
    batch_size = int(sys.argv[3])
    num_epochs = int(sys.argv[4])
else:
    print "USING DEFAULT PARAMETERS"
    train_samples = 400
    test_samples = 100
    batch_size = 64
    num_epochs=10

print "********************************************"
print "TRAIN SAMPLES:       ", train_samples
print "TEST SAMPLES:        ", test_samples
print "BATCH SIZE:          ", batch_size
print "NUMBER OF EPOCHS:    ", num_epochs
print "********************************************"

timestr = time.strftime("%Y%m%d-%H%M%S")
modelname = timestr + str(train_samples) + str(test_samples)
print "This Model Name:" + modelname

##Defs

# Return the solutions for given keys
def process_keys(sols, keys):
    count = len(keys)
    tmp_arr_37 = np.zeros(shape=(count,37))
    tmp_arr_3 = np.zeros(shape=(count, 3))
    int_arr = []
    for c, key in enumerate(keys):
        # Get the 37 values associated with each key
        tmp_arr_37[c,:] = sols.get(key, None)
        tmp_arr_3[c,0] = tmp_arr_37[c,0]
        tmp_arr_3[c,1] = tmp_arr_37[c,1]
        tmp_arr_3[c,2] = tmp_arr_37[c,2]
    # Turn them all into integers from 0-2
    for i in range(len(keys)):
        int_arr.append(np.argmax(tmp_arr_3[i,:]))
    #print "int_arr:",int_arr
    #one-hot encode
    #arr = to_categorical(int_arr, num_classes = None)
    return int_arr


# Returns the paths of all files in the input directory

def get_paths(directory, train_size, test_size):
    dirs = [f for f in os.listdir(directory)]
    #return [f for f in os.listdir(directory)]
    print "trn+tst:", train_size + test_size
    dirs = random.sample(set(dirs), train_size + test_size)
    train_dirs = dirs[:train_size]
    test_dirs = dirs[train_size:]
    print "trn:",len(train_dirs)
    print "tst:",len(test_dirs)
    print "********************************************"
    return train_dirs, test_dirs
"""
def get_paths(directory, size=None):
    dirs = [f for f in os.listdir(directory)]
    #return [f for f in os.listdir(directory)]
    if size != None:
        print "Retrieving ", size, "samples from ", directory, "with population = ", len(dirs)
        dirs = random.shuffle(dirs)
        dirs = dirs[0:(size-1)]
        print "length of directory array:   ", len(dirs)
    return dirs
"""

# Imports the solution csv file and returns them as an array
def get_solutions(filepath):
    all_solutions = {}
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for i, line in enumerate(reader):
            all_solutions[line[0]] = [float(x) for x in line[1:]]
    return all_solutions

"""
# Makes the vgg16 model
def ConvBlock(layers, model, filters, activation_fn):

    for i in range(layers):
        model.add(ZeroPadding2D((1,1)))  # zero padding of size 1
        model.add(Conv2D(filters, (3, 3), activation=activation_fn))  # 3x3 filter size
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format="channels_first"))

def FCBlock(model, activation_fn):

    model.add(Dense(4096, activation=activation_fn, kernel_initializer='normal'))
    model.add(Dropout(0.5))

def get_model(activation_fn):

    model = Sequential()
    model.add(Lambda(lambda x : x, input_shape=(106,106,3)))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

    ConvBlock(2, model, 64, activation_fn)
    ConvBlock(2, model, 128, activation_fn)
    ConvBlock(3, model, 256, activation_fn)
    ConvBlock(3, model, 512, activation_fn)
    ConvBlock(3, model, 512, activation_fn)

    model.add(Flatten())
    FCBlock(model, activation_fn)
    FCBlock(model, activation_fn)

    model.add(Dense(3, activation = 'softmax'))
    return model
"""

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(106,106,3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format="channels_first"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format="channels_first"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format="channels_first"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format="channels_first"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format="channels_first"))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model



#Keras imports
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda, Reshape
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.models import load_model
#new
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalMaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils.np_utils import to_categorical
from keras import metrics


"""
def get_callbacks(filepath, patience=2):
   es = EarlyStopping('val_loss', patience=10, mode="min")
   msave = ModelCheckpoint(filepath, save_best_only=True)
   return [es, msave]
"""

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc= []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))


##Code:

print "GET PATHS"
# Get the image names + paths
train_names = []
test_names = []
train_names, test_names = get_paths(base_dir + img_dir, train_samples, test_samples)
#test_names = get_paths(base_dir + img_dir + valid_dir, test_samples)


# Create the path names from the image names
print "RETRIEVING IMAGE PATHS"
train_paths = []
test_paths = []
for i in range(len(train_names)):
    train_paths.append(base_dir + img_dir + train_names[i])
for i in range(len(test_names)):
    test_paths.append(base_dir + img_dir + test_names[i])

# Cutting the last four characters from the names to get only the keys
for i in range(len(train_names)):
    train_names[i]=train_names[i][:-4]
for i in range(len(test_names)):
    test_names[i]=test_names[i][:-4]

print "GET LABELS"
#get labels for train + validation sets
sols = get_solutions(base_dir + "training_solutions_rev1.csv")
train_lab = process_keys(sols, train_names)
test_lab = process_keys(sols, test_names)
print "TRAIN LABEL 1:", train_lab[0]

# Downscales images for a given path to an image. Returns 3 dim array of pixel values
test_count=0
def val_batch_generator(test_paths, test_lab):
    #Not sure if this will work but run and see
    #tmp_img = np.zeros(shape=(106,106,3))
    while True:
        sols=[]
        imgs=[]
        for i, thispath in enumerate(test_paths):
            tmp_sol = test_lab[i]
            sols.append(tmp_sol)
            #print "train solution before reshape",tmp_sol
            #tmp_sol = np.reshape(tmp_sol,(1,3))
            #print "train solution:",tmp_sol
            tmp_img = cv2.imread(thispath)     #read in image
            #print tmp_img
            tmp_img = tmp_img[106:318, 106:318] #crop 424x424 -> 212x212. Centred
            tmp_img = cv2.resize(tmp_img, (106,106), interpolation = cv2.INTER_AREA)  #resizing
            imgs.append(tmp_img)
            #tmp_img=np.reshape(tmp_img,(1,3,106,106))
            global test_count
            test_count+=1
            if (i+1)%batch_size==0:
                #tmp_sol.reshape(batch_size,3)
                sols = to_categorical(sols, num_classes=3)
                #print "Yielding"
                imgs = np.asarray(imgs)
                sols = np.asarray(sols)
                #print "imgs shape", imgs.shape
                #print "sols shape", sols.shape
                #convert to numpy
                yield (imgs, sols)
                if(i>(len(test_lab)-batch_size)):
                    "test break\n"
                    break
                sols=[]
                imgs=[]

#Need a generator, which needs an image array, which needs labels
#Rotate images - generator
#Needs input data array [1,3,106,106] and labels
train_count=0
"""
def batch_generator(train_paths, train_lab):
    #Not sure if this will work but run and see
    tmp_img = np.zeros(shape=(106,106,3))
    while True:
        for i, thispath in enumerate(train_paths):
            tmp_sol = train_lab[i]

            tmp_sol = to_categorical(tmp_sol, num_classes=3)
            #print "train solution before reshape",tmp_sol
            tmp_sol = np.reshape(tmp_sol,(1,3))
            #print "train solution:",tmp_sol
            img = cv2.imread(thispath)     #read in image
            #print "TRAIN PATH: ", path
            img = img[106:318, 106:318] #crop 424x424 -> 212x212. Centred
            tmp_img = cv2.resize(img, (106,106), interpolation = cv2.INTER_AREA)    #resizing
            tmp_img=np.reshape(tmp_img,(1,3,106,106))
            global train_count
            train_count+=1

            yield (tmp_img, tmp_sol)
"""

def batch_generator(train_paths, train_lab):
    #Not sure if this will work but run and see
    #tmp_img = np.zeros(shape=(106,106,3))
    while True:
        sols=[]
        imgs=[]
        for i, thispath in enumerate(train_paths):
            tmp_sol = train_lab[i]
            sols.append(tmp_sol)
            #print "train solution before reshape",tmp_sol
            #tmp_sol = np.reshape(tmp_sol,(1,3))
            #print "train solution:",tmp_sol
            tmp_img = cv2.imread(thispath)     #read in image
            #print "TRAIN PATH: ", path
            tmp_img = tmp_img[106:318, 106:318] #crop 424x424 -> 212x212. Centred
            tmp_img = cv2.resize(tmp_img, (106,106), interpolation = cv2.INTER_AREA)  #resizing
            imgs.append(tmp_img)
            #tmp_img=np.reshape(tmp_img,(1,3,106,106))
            global train_count
            train_count+=1
            if (i+1)%batch_size==0:
                #tmp_sol.reshape(batch_size,3)
                sols = to_categorical(sols, num_classes=3)
                #print "Yielding"
                imgs = np.asarray(imgs)
                sols = np.asarray(sols)
                #print "imgs shape", imgs.shape
                #print "sols shape", sols.shape
                #convert to numpy
                yield (imgs, sols)
                if(i>(len(train_lab)-batch_size)):
                    "train break"
                    break
                sols=[]
                imgs=[]


filepath = base_dir + "data/" + str(modelname) + "model_weights_" + str(train_samples) + ".hdf5"
checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)

history = LossHistory()
#callbacks = get_callbacks(filepath, patience=10)

print "********************************************"
print "Steps per epoch:", int(len(train_paths)/batch_size)
print "Batch size:", batch_size
print "Training samples per epoch:", len(train_paths)
print "Testing samples per epoch:", len(test_paths)
print "TRAIN MODEL..."
print "********************************************"

optimizer = RMSprop(lr=learning_rate)
mymodel=VGG_16()
#mymodel.compile(loss=loss_fn, optimizer=optimizer,
                #metrics=[metrics.mae, metrics.categorical_accuracy])
mymodel.compile(loss=loss_fn, optimizer=optimizer,
                metrics=['accuracy'])


hist = mymodel.fit_generator(
            batch_generator(train_paths, train_lab),
            steps_per_epoch=int(len(train_paths)/batch_size),
            epochs=num_epochs,
            verbose=2,
            validation_data=val_batch_generator(test_paths, test_lab),
            validation_steps=int(len(test_paths)/batch_size),
            shuffle = True,
            callbacks=[history, checkpointer])



print "********************************************"
# ### Plot training/validation loss
print "WRITING DATA TO FILES"
plotpath = "/home/keegansmith/keras/ML/plots/"
with open(plotpath + "data/" + modelname + "_loss.txt", 'w') as f:
    for line in hist.history['loss']:
        f.write(str(line)+"\n")
with open(plotpath + "data/" + modelname + "_val_loss.txt", 'w') as f:
    for line in hist.history['val_loss']:
        f.write(str(line)+"\n")
with open(plotpath + "data/" + modelname + "_epochs.txt", 'w') as f:
    for line in hist.epoch:
        f.write(str(line)+"\n")
with open(plotpath + "data/" + modelname + "_acc.txt", 'w') as f:
    for line in hist.history['acc']:
        f.write(str(line)+"\n")
with open(plotpath + "data/" + modelname + "_val_acc.txt", 'w') as f:
    for line in hist.history["val_acc"]:
        f.write(str(line)+"\n")

print "********************************************"
print "Total images trained on:", train_count
print "Total images tested on:", test_count
print "Script complete in " + str((datetime.now()-start))
