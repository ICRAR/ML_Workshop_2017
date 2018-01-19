# # Galaxy Zoo with keras

"""
Command line arguments:
1 - Learning Rate. Default = 1e-6
2 - Loss function. Default = mean squared error
3 - Activation function. Default = sigmoid
"""

import os
import numpy as np
import glob
import cv2
from datetime import datetime
import random
import string
import time
import sys

data_path = '/home/keegansmith/keras/data/'
base_path = '/home/keegansmith/keras/'

from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda, Reshape
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils.np_utils import to_categorical

# Begin timing
start = datetime.now()


#Initialize arguments
if len(sys.argv)==4:
    learning_rate = float(sys.argv[1])
    loss_fn = str(sys.argv[2])
    activation_fn = str(sys.argv[3])
else:
    #defualts
    "USING DEFAULT PARAMETERS"
    learning_rate = 1e-6
    loss_fn = 'mean_squared_error'
    activation_fn = 'sigmoid'


print "LEARNING RATE:           " + str(learning_rate)
print "LOSS FUNCTION:           " + str(loss_fn)
print "ACTIVATION FUNCTION:     " + str(activation_fn)


#Some defs
def ConvBlock(layers, model, filters):
    """
    Create a layered Conv/Pooling block
    """
    for i in range(layers):
        model.add(ZeroPadding2D((1,1)))  # zero padding of size 1
        model.add(Conv2D(filters, (3, 3), activation='relu'))  # 3x3 filter size
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format="channels_first"))

def FCBlock(model):

    """
    Fully connected block with ReLU and dropout
    """
    model.add(Dense(4096, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.5))

def VGG_16():
    """
    Implement VGG16 architecture
    """
    model = Sequential()
    model.add(Lambda(lambda x : x, input_shape=(3,106,106)))
    #model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

    ConvBlock(2, model, 64)
    ConvBlock(2, model, 128)
    ConvBlock(3, model, 256)
    ConvBlock(3, model, 512)
    ConvBlock(3, model, 512)

    model.add(Flatten())
    FCBlock(model)
    FCBlock(model)

    model.add(Dense(37, input_dim=106*106*3, activation = activation_fn))
    return model



# Compile
optimizer = RMSprop(lr=1e-6)
model = VGG_16()
model.compile(loss=str(loss_fn), metrics=['accuracy'], optimizer=optimizer)
#model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer=optimizer)

#Make a unique name for the model with a stimestamp
timestr = time.strftime("%Y%m%d-%H%M%S")
modelname = timestr + "_" + loss_fn + "_" + activation_fn + "_" + str(learning_rate)
print "This Model Name:" + modelname
model.save(data_path + modelname + ".hdf5")


# In[49]:

import csv
from random import shuffle
#from scipy.misc import imresize

class data_getter:
    """
    Creates a class for handling train/valid/test data paths,
    training labels and image IDs.
    Useful for switching between sample and full datasets.
    """
    def __init__(self, path):
        self.path = path
        self.train_path = path + "train"
        self.val_path = path + "valid"
        self.test_path = path + "test"

        def get_paths(directory):
            return [f for f in os.listdir(directory)]

        self.training_images_paths = get_paths(self.train_path)
        self.validation_images_paths = get_paths(self.val_path)
        self.test_images_paths = get_paths(self.test_path)

        def get_all_solutions():
        ### Import solutions file and load into self.solutions

            all_solutions = {}
            with open(base_path + 'training_solutions_rev1.csv', 'r') as f:
                reader = csv.reader(f, delimiter=",")
                next(reader)
                for i, line in enumerate(reader):
                    all_solutions[line[0]] = [float(x) for x in line[1:]]
            return all_solutions

        self.all_solutions = get_all_solutions()

    def get_id(self,fname):
        return fname.replace(".jpg","").replace("data","")

    def find_label(self,val):
        return self.all_solutions[val]

# fetcher = data_getter('data/sample/')
fetcher = data_getter(data_path + "images/training/")
print "FETCHER:"
#print fetcher
print(fetcher.train_path)



def process_images(paths):
    """
    Import image at 'paths', decode, centre crop and prepare for batching.
    """
    count = len(paths)
    #arr = np.zeros(shape=(count,106,106,3))
    arr = np.zeros(shape=(count,3,106,106))
    #print "array shape: " + str(arr.shape)
    for c, path in enumerate(paths):
        img = cv2.imread(path)     #read in image
        img = img[106:318, 106:318] #crop 424x424 -> 212x212. Centred
        img = cv2.resize(img, (106,106), interpolation = cv2.INTER_AREA)    #resizing
        arr[c,0] = img[:,:,0]
        arr[c,1] = img[:,:,1]
        arr[c,2] = img[:,:,2]
    return arr




"""
## Print some before/after processing images

#print fetcher.training_images_paths
process_images([fetcher.train_path + '/' + fetcher.training_images_paths[100]])#
im = cv2.imread(fetcher.train_path + '/' + fetcher.training_images_paths[0])
print "IMAGE SHAPE:"
print(im.shape)#
import matplotlib.pyplot as plt
plt.imshow(im)
plt.show()
im = im[106:318, 106:318] #crop 424x424 -> 212x212. Centred
im = cv2.resize(im, (106,106), interpolation = cv2.INTER_AREA)
print "IMAGE SHAPE:"
print(im.shape)#
plt.imshow(im)
plt.show()
"""



# Create generator that yields (current features X, current labels y)
def BatchGenerator(getter):
    while 1:
        for f in getter.training_images_paths:
            X_train = process_images([getter.train_path + '/' + fname for fname in [f]])
            id_ = getter.get_id(f)
            y_train = np.array(getter.find_label(id_))
            #y_train = to_categorical(y_train, 37)
            y_train = np.reshape(y_train,(1,37))
            yield (X_train, y_train)

def ValBatchGenerator(getter):
    while 1:
        for f in getter.validation_images_paths:
            X_train = process_images([getter.val_path + '/' + fname for fname in [f]])
            id_ = getter.get_id(f)
            y_train = np.array(getter.find_label(id_))
            #y_train = to_categorical(y_train, 37)
            y_train = np.reshape(y_train,(1,37))
            yield (X_train, y_train)



from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.models import load_model

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

#early_stopping = EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')
history = LossHistory()


from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath=data_path + modelname + ".hdf5", verbose=1, save_best_only=True)


batch_size = 32
num_epochs = 50
steps_to_take = int(len(fetcher.training_images_paths)/batch_size)
val_steps_to_take = int(len(fetcher.validation_images_paths)/batch_size)
print "BATCH SIZE:"
print batch_size
print "NUMBER OF EPOCHS:"
print num_epochs
print "Steps_to_take"
print steps_to_take
print "Val_steps_to_take"
print val_steps_to_take
print "LOADING MODEL"

model = load_model(data_path + modelname + ".hdf5")
hist = model.fit_generator(BatchGenerator(fetcher),
                    steps_per_epoch=steps_to_take,
                    epochs=50,
                    validation_data=ValBatchGenerator(fetcher),
                    verbose=2,
                    callbacks=[history,checkpointer],
					validation_steps=val_steps_to_take
                   )


# ### Plot training/validation loss
print "WRITING DATA TO FILES"
plotpath = "/home/keegansmith/keras/plots/"
with open(plotpath + "data/" + modelname + "_loss.txt", 'w') as f:
    for line in hist.history['loss']:
        f.write(str(line)+"\n")
with open(plotpath + "data/" + modelname + "_val_loss.txt", 'w') as f:
    for line in hist.history['val_loss']:
        f.write(str(line)+"\n")
with open(plotpath + "data/" + modelname + "_epochs.txt", 'w') as f:
    for line in hist.epoch:
        f.write(str(line)+"\n")
with open(plotpath + "data/" + modelname + "_val_acc.txt", 'w') as f:
    for line in hist.history["val_acc"]:
        f.write(str(line)+"\n")
with open(plotpath + "data/" + modelname + "_acc.txt", 'w') as f:
    for line in hist.history["acc"]:
        f.write(str(line)+"\n")

# Find the lowest value for the loss and write the results
temp_val_loss = hist.history["val_loss"]
answer_dex = temp_val_loss.index(max(temp_val_loss))

with open(base_path + "solutions/" + modelname + ".txt", 'w') as f:
    f.write("lR: " + str(learning_rate) + "\n"
            + "Loss fn: " + loss_fn + "\n"
            + "Activation fn: " + activation_fn + "\n\n")
    f.write("loss: " + str(hist.history["loss"][answer_dex]) + "\n"
            "val_loss: " + str(hist.history["val_loss"][answer_dex]) + "\n"
            "acc: " + str(hist.history["acc"][answer_dex]) + "\n"
            "val_acc: " + str(hist.history["val_acc"][answer_dex]))


# ### Model Predict

# Load best model weights
from keras.models import load_model
print "LOADING MODEL"
model = load_model(data_path + modelname+".hdf5")
print "MODEL LOADED"


def TestBatchGenerator(getter):
	while 1:
		print "TESTING..."
		for f in getter.validation_images_paths:
			X_train = process_images([getter.val_path + '/' + fname for fname in [f]])
			yield (X_train)


predictions = model.predict_generator(TestBatchGenerator(fetcher), steps = len(fetcher.validation_images_paths), max_queue_size = 32)

predictions.shape

header = open(base_path + 'all_zeros_benchmark.csv','r').readlines()[0]

with open(base_path + modelname + 'submission_1.csv','w') as outfile:
    outfile.write(header)
    for i in range(len(fetcher.validation_images_paths)):
        id_ = (fetcher.get_id(fetcher.validation_images_paths[i]))
        pred = predictions[i]
        outline = id_ + "," + ",".join([str(x) for x in pred])
        outfile.write(outline + "\n")


# Finish timing
print "Script complete in " + str((datetime.now()-start))
