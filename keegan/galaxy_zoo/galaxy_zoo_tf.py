#Galaxy zoo with tensorflow

import os
import numpy as np
import glob
import cv2
import tensorflow as tf
import csv
from datetime import datetime

# Begin timing
start = datetime.now()


def model(X, w_h, w_h2, w_o, drop_input, drop_hidden):
    # Add layer name scopes for better graph visualization
    with tf.name_scope("layer1"):
        X = tf.nn.dropout(X, drop_input)
        h = tf.nn.relu(tf.matmul(X, w_h))
    with tf.name_scope("layer2"):
        h = tf.nn.dropout(h, drop_hidden)
        h2 = tf.nn.relu(tf.matmul(h, w_h2))
    with tf.name_scope("layer3"):
        h2 = tf.nn.dropout(h2, drop_hidden)
    return tf.matmul(h2, w_o)


# Downscales images for a given path to an image. Returns 3 dim array of pixel values
def process_images(paths):

    count = len(paths)
    #arr = np.zeros(shape=(count,106,106,3))
    arr = np.zeros(shape=(count,3*106*106))
    #print "array shape: " + str(arr.shape)
    for c, path in enumerate(paths):
        img = cv2.imread(path)     #read in image in 3-channel colour
        img = img[106:318, 106:318] #crop 424x424 -> 212x212. Centred
        img = cv2.resize(img, (106,106), interpolation = cv2.INTER_AREA)    #resizing
        count=0
        for i in range(106):
            for j in range(106):
                for k in range(3):
                    arr[c,count]=img[i,j,k]
                    count+=1
        """
    count = len(paths)
    #arr = np.zeros(shape=(count,106,106,3))
    #print count
    arr = np.zeros(shape=(count,3,106,106))
    #print "array shape: " + str(arr.shape)
    for c, path in enumerate(paths):
        #print path
        img = cv2.imread(path)     #read in image
        img = img[106:318, 106:318] #crop 424x424 -> 212x212. Centred
        img = cv2.resize(img, (106,106), interpolation = cv2.INTER_AREA)    #resizing
        arr[c,0] = img[:,:,0]
        arr[c,1] = img[:,:,1]
        arr[c,2] = img[:,:,2]
    """
    return arr


# Return the solutions for given keys
def process_keys(sols, keys):
    count = len(keys)
    arr = np.zeros(shape=(count,37))

    for c, key in enumerate(keys):
        # Get the 37 values associated with each key
        arr[c,:] = sols.get(key, None)
    return arr


# Returns the paths of all files in the input directory
def get_paths(directory):
    return [f for f in os.listdir(directory)]


# Imports the solution csv file and returns them as an array
def get_solutions(filepath):
    all_solutions = {}
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for i, line in enumerate(reader):
            all_solutions[line[0]] = [float(x) for x in line[1:]]
    return all_solutions
#'training_solutions_rev1.csv'  --  solutions file


# Used to initialize the weights randomly via gassian distribution
def init_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)


# Define some stuff:
img_dir = "/home/keegansmith/keras/data/images/training/"
base_dir = "/home/keegansmith/keras/"
epochs = 25
batch_size=200

# Get the image names
train_names = []
train_names = get_paths(img_dir + "train/" )
test_names = []
test_names = get_paths(img_dir + "valid/")


# Create the path names from the image names
print "RETRIEVING IMAGE PATHS"
train_paths = []
test_paths = []
for i in range(len(train_names)):
    train_paths.append(img_dir + "train/" + train_names[i])
for i in range(len(test_names)):
    test_paths.append(img_dir + "valid/" + test_names[i])

# Cutting the last four characters from the names to get only the keys
for i in range(len(train_names)):
    train_names[i]=train_names[i][:-4]
for i in range(len(test_names)):
    test_names[i]=test_names[i][:-4]

"""
# Get input data
tr_im, tr_la, te_im, te_la = [], [], [], []
tr_im = get_paths(img_dir + "/train")
te_im = get_paths(img_dir + "/valid")
"""

# Creating placeholders for images + labels
# 11236 = 106^2. Multiply by 3 for each channel
X = tf.placeholder(tf.float32, [None, 106*106*3], name="X")
label = tf.placeholder(tf.float32, [None, 37], name="Label")


# Setting up weights
print "APPLYING INITIAL WEIGHTS"
weight_h1 = init_weights([33708, 6000], "weight_h1")
weight_h2 = init_weights([6000, 6000], "weight_h2")
weight_out = init_weights([6000, 37], "weight_out")


# Add histogram summaries for weights
tf.summary.histogram("weight_h1_summ", weight_h1)
tf.summary.histogram("weight_h2_summ", weight_h2)
tf.summary.histogram("weight_out_summ", weight_out)


# Add dropout to input and hidden layers
#print "APPLYING DROPOUTS"
drop_input = tf.placeholder(tf.float32, name="drop_input")
drop_hidden = tf.placeholder(tf.float32, name="drop_hidden")


# Create Model
mymod = model(X, weight_h1, weight_h2, weight_out, drop_input, drop_hidden)


# Create cost function
with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mymod, labels=label))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    # Add scalar summary for cost tensor
    tf.summary.scalar("cost", cost)


# Measure accuracy
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(label, 1), tf.argmax(mymod, 1)) # Count correct predictions
    acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # Cast boolean to float to average
    # Add scalar summary for accuracy tensor
    tf.summary.scalar("accuracy", acc_op)


# Acquire the solutions from the csv file
print "ACQUIRING SOLUTIONS"
sols = get_solutions(base_dir + "training_solutions_rev1.csv")


# Initialize the test images and keys
print "PROCESSING LABELS"
#train_im = process_images(train_paths)
train_lab = process_keys(sols, train_names)
test_im = process_images(test_paths)
test_lab = process_keys(sols, test_names)
train_batch = []
test_batch = []

# Create a session
print "LAUNCHING SESSION"
with tf.Session() as sess:
    # First make a log writer
    writer = tf.summary.FileWriter("tb_logs/nn_logs")
    merged = tf.summary.merge_all()

    # Initialize_all_variables
    tf.global_variables_initializer().run()

    # Train the model
    for i in range(epochs):
        print "BEGINNING EPOCH " + str(i+1)
        for start, end in zip(range(0, len(train_paths), batch_size), range(batch_size, len(train_paths)+1, batch_size)):
            """ For example: if batch_size = 10, (start, end) = (0,10) -> (10, 20) -> (20, 30) etc """
            print "start, end:  ", start, end

            #Generate a batch of images
            #print train_paths[start:end]
            train_batch = process_images(train_paths[start:end])

            print "TRAINING"
            sess.run(train_op, feed_dict={X: train_batch, label: train_lab[start:end], drop_input: 0.8, drop_hidden: 0.5})
            print "FINISHED BATCH"


        print "FINISHED EPOCH"
        summary, acc = sess.run([merged, acc_op], feed_dict={X: test_im, label: test_lab,
                                          drop_input: 1.0, drop_hidden: 1.0})
        writer.add_summary(summary, i)  # Write summary
        print "EPOCH: ", (i+1), "     ACC: ", acc                   # Report the accuracy


print "Script complete in " + str(datetime.now()-start)

"""
# Create a session
with tf.Session() as sess:
    # Create a log writer. run 'tensorboard --logdir=./logs/nn_logs'
    writer = tf.summary.FileWriter("./logs/nn_logs", sess.graph) # for 0.8
    merged = tf.summary.merge_all()

    # Initialize all variables
    tf.initialize_all_variables().run()

    # Train the  model
    for i in range(100):
        for start, end in zip(range(0, len(tr_im), 128), range(128, len(tr_im)+1, 128)):
            sess.run(train_op, feed_dict={X: tr_im[start:end], Y: tr_la[start:end],
                                          drop_input: 0.8, drop_hidden: 0.5})
        summary, acc = sess.run([merged, acc_op], feed_dict={X: te_im, Y: te_la,
                                          drop_input: 1.0, drop_hidden: 1.0})
        writer.add_summary(summary, i)  # Write summary
        print(i, acc)
"""
