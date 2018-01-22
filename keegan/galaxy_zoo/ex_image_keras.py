# Takes an example of an image from the galaxy_zoo_keras.py
# Compares the computed answer and the actual solution

import os
import numpy as np
import glob
import cv2
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import sys

if len(sys.argv)==2:
    file_precursor=sys.argv[1]
else:
    file_precursor=""


def get_val_ans():
### Import solutions file and load into self.solutions
    all_solutions = {}
    with open(file_precursor + 'submission_1.csv', 'r') as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for i, line in enumerate(reader):
            all_solutions[line[0]] = [float(x) for x in line[1:]]
    return all_solutions

def get_all_solutions():
### Import solutions file and load into self.solutions
    all_solutions = {}
    with open('training_solutions_rev1.csv', 'r') as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for i, line in enumerate(reader):
            all_solutions[line[0]] = [float(x) for x in line[1:]]
    return all_solutions

### Deletes contents of a folder
def delete_dir_contents(dir_path):
    files = glob.glob(dir_path)
    for f in files:
        os.remove(f)


# Find an example image and compare to the solution
img_dir = "data/images/training/valid/"
img_paths = []
val_answers = get_val_ans()
all_sols = get_all_solutions()
#print len(val_answers.keys())
#print len(all_sols.keys())

font = {'family': 'serif',
        'color':  'white',
        'weight': 'normal',
        'size': 12,
        }

#Delete contents of folder first
delete_dir_contents("ex_plots/*")

#keys = []
#for i in val_answers:
#    keys.append(i)

for i in range(9):
    #key = next(iter(val_answers))

    key = random.choice(val_answers.keys())
    #key = random.choice(keys)
    answer = []
    answer = val_answers.get(key, 0)

    #Now find the actual answer from the solutions dictionary
    solution = []
    solution = all_sols.get(key, 0)

    # Now find the indexes with the highest values
    answer_dex = answer.index(max(answer))
    solution_dex = solution.index(max(solution))

    img_paths.append(img_dir + str(key)+ ".jpg")

    print "IMAGE ID: " + str(key)
    print answer_dex, answer[answer_dex]
    print solution_dex, solution[solution_dex]
    print "---------------------------------- \n"

    img = mpimg.imread(img_dir + str(key)+ ".jpg")
    plt.imshow(img)
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)
    plt.text(380,30, str(i+1), fontdict=font)
    plt.text(30,30,"Galaxy ID: " + str(key), fontdict=font)
    plt.text(30,60,"Predicted Class -   " + str(answer_dex), fontdict=font)
    plt.text(30,90,"True Class -        " + str(solution_dex), fontdict=font)
    plt.savefig("ex_plots/"+str(i+1) + "_" +  str(key) + "_infoplot.jpg")
    plt.close()
