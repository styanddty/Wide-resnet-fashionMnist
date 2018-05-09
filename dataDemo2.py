import h5py
import os
import numpy as np

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

file = h5py.File('processed_data.h5','r+')

#Retrieves all the preprocessed training and validation\testing data from a file

X_train = file['X_train'][...]
Y_train = file['Y_train'][...]
X_val = file['X_val'][...]
Y_val = file['Y_val'][...]
X_test = file['X_test'][...]
Y_test = file['Y_test'][...]

# Unpickles and retrieves class names and other meta informations of the database
classes = unpickle('cifar-10-batches-py/batches.meta') #keyword for label = label_names

print("Training sample shapes (input and output): "+str(X_train.shape)+" "+str(Y_train.shape))
print("Validation sample shapes (input and output): "+str(X_val.shape)+" "+str(Y_val.shape))
print("Testing sample shapes (input and output): "+str(X_test.shape)+" "+str(Y_test.shape))

# Creates nested list. The outer list will list all the classess (0-9). And each of the classes represent the inner list which list all
# training data that belongs to that class. I used list because it is easy to keep on adding dynamically. Ndarrays may have needed
# a predifined shape

classes_num = len(classes['label_names'])  # classes_num = no. of classes

# Here, I am creating a special variable X_train_F which is basically a nested list.
# The outermost list of X_train_F will be a list of all the class values (0-9 where each value correspond to a class name)
# Each elements (class values) of the outermost list is actually also a list; a list of all the example data belonging
# to the particular class which corresponds to class value under which the data is listed.

X_train_F = []

for i in xrange(0, classes_num):
    X_train_F.append([])

for i in xrange(0, len(X_train)):
    l = np.argmax(Y_train[i])  # l for label (in this case it's basically the index of class value elemenmts)
    # (Y_train is one hot encoded. Argmax returns the index for maximum value which should be 1 and
    # that index should indicate the value)
    X_train_F[l].append(X_train[i])

# for i in xrange(classes_num):
#     print "X_train_F[", classes['label_names'][i], "] = ", len(X_train_F[i])
import matplotlib.pyplot as plt
from scipy.misc import toimage
from scipy.misc import imresize
# %matplotlib inline
#function for showing pictures in grid along with labels

def picgrid(X_train,Y_train, filename = "demoHello"):
    gray = 0
    plt.figure(figsize=(7,7))
    ax=[]
    for i in xrange(0,25):
        img = toimage(X_train[i])
        ax.append(plt.subplot(5,5,i+1))
        ax[i].set_title( classes['label_names'][np.argmax(Y_train[i])],y=-0.3)
        ax[i].set_axis_off()
        if gray==0:
            plt.imshow(img)
        else:
            plt.imshow(img,cmap='gray')
    plt.subplots_adjust(hspace=0.3)
    plt.axis('off')
    # plt.show()
    plt.savefig(filename)
# picgrid(X_train, Y_train)

import random

smoothing_factor = 0.1  # for label smoothing


def create_batches(batch_size, classes_num):
    s = int(batch_size / classes_num)  # s denotes samples taken from each class to create the batch.
    print "s=", s
    no_of_batches = int(len(X_train) / batch_size)

    shuffled_indices_per_class = []
    for i in xrange(0, classes_num):
        temp = np.arange(len(X_train_F[i]))
        np.random.shuffle(temp)
        shuffled_indices_per_class.append(temp)

    batches_X = []
    batches_Y = []

    for i in xrange(no_of_batches):

        shuffled_class_indices = np.arange(classes_num)
        np.random.shuffle(shuffled_class_indices)

        batch_Y = np.zeros((batch_size, classes_num), np.float32)
        batch_X = np.zeros((batch_size, 32, 32, 3), np.float32)

        for index in xrange(0, classes_num):
            class_index = shuffled_class_indices[index]
            for j in xrange(0, s):
                batch_X[(index * s) + j] = X_train_F[class_index][shuffled_indices_per_class[class_index][
                    i * s + j]]  # Assign the s chosen random samples to the training batch
                batch_Y[(index * s) + j][class_index] = 1
                batch_Y[(index * s) + j] = (1 - smoothing_factor) * batch_Y[
                    (index * s) + j] + smoothing_factor / classes_num
                # print batch_Y[(index * s) + j]


        rs = batch_size - s * classes_num  # rs denotes no. of random samples from random classes to take
        # in order to fill the batch if batch isn't divisble by classes_num
        # fill the rest of the batch with random data
        rand = random.sample(np.arange(len(X_train)), rs)
        j = 0
        for k in xrange(s * classes_num, batch_size):
            batch_X[k] = X_train[int(rand[j])]
            batch_Y[k] = Y_train[int(rand[j])]
            batch_Y[k] = (1 - smoothing_factor) * batch_Y[k] + smoothing_factor / classes_num
            j += 1

        batches_X.append(batch_X)
        batches_Y.append(batch_Y)

    return batches_X, batches_Y


batches_X, batches_Y = create_batches(64, classes_num)  # A demo of the function at work

# Since each batch will have almost equal no. of cases from each class, no batch should be biased towards some particular classes

sample = random.randint(0, len(batches_X))
print "Sample arranged images in a batch: "
picgrid(batches_X[sample], batches_Y[sample], "demohello2")


def random_crop(img):
    # result = np.zeros_like((img))
    c = np.random.randint(0, 5)
    if c == 0:
        crop = img[4:32, 0:-4]
    elif c == 1:
        crop = img[0:-4, 0:-4]
    elif c == 2:
        crop = img[2:-2, 2:-2]
    elif c == 3:
        crop = img[4:32, 4:32]
    elif c == 4:
        crop = img[0:-4, 4:32]

    # translating cropped position
    # over the original image
    c = np.random.randint(0, 5)
    if c == 0:
        img[4:32, 0:-4] = crop[:]
    elif c == 1:
        img[0:-4, 0:-4] = crop[:]
    elif c == 2:
        img[2:-2, 2:-2] = crop[:]
    elif c == 3:
        img[4:32, 4:32] = crop[:]
    elif c == 4:
        img[0:-4, 4:32] = crop[:]

    return img
plt.figure(figsize=(6,6))
ax=[]
for i in xrange(0, 16, 2):
    img = toimage(random_crop(X_train[i]))
    ax.append(plt.subplot(4,4,i+1))
    ax[i].set_title( classes['label_names'][np.argmax(Y_train[i])],y=-0.3)
    ax[i].set_axis_off()
    # img2 = random_crop(img)
    plt.imshow(img)

    img = toimage(X_train[i])
    ax.append(plt.subplot(4, 4, i + 2))
    ax[i+1].set_title(classes['label_names'][np.argmax(Y_train[i])], y=-0.3)
    ax[i+1].set_axis_off()
    # img2 = random_crop(img)
    plt.imshow(img)

plt.subplots_adjust(hspace=0.3)
plt.axis('off')
# plt.show()
plt.savefig("dataDemoCrop3")
# print "say hello to all"
# # picgrid(X_train, Y_train)

def augment_batch(batch_X):  # will be used to modify images realtime during training (real time data augmentation)

    aug_batch_X = np.zeros((len(batch_X), 32, 32, 3))

    for i in xrange(0, len(batch_X)):

        hf = np.random.randint(0, 2)

        if hf == 1:  # hf denotes horizontal flip. 50-50 random chance to apply horizontal flip on images,
            batch_X[i] = np.fliplr(batch_X[i])

        # Remove the below cropping to apply random crops. But before that it's better to implement something like mirror padding
        # or any form of padding to increase the dimensions beforehand.

        c = np.random.randint(0, 3)
        if c == 1:
            # one in a three chance for cropping
            # randomly crop 28x28 portions and translate it.
            aug_batch_X[i] = random_crop(batch_X[i])
        else:
            aug_batch_X[i] = batch_X[i]

    return aug_batch_X


aug_batches_X = []
for batch in batches_X:
    aug_batch_X = augment_batch(batch)
    aug_batches_X.append(aug_batch_X)

print "Sample batch training images after augmentation:"
picgrid(aug_batches_X[sample], batches_Y[sample], "demohello3")


def shuffle_batch(batch_X, batch_Y):
    shuffle = random.sample(np.arange(0, len(batch_X), 1, 'int'), len(batch_X))
    shuffled_batch_X = []
    shuffled_batch_Y = []

    for i in xrange(0, len(batch_X)):
        shuffled_batch_X.append(batch_X[int(shuffle[i])])
        shuffled_batch_Y.append(batch_Y[int(shuffle[i])])

    shuffled_batch_X = np.array(shuffled_batch_X)
    shuffled_batch_Y = np.array(shuffled_batch_Y)

    return shuffled_batch_X, shuffled_batch_Y


s_batches_X = []
s_batches_Y = []
for i in xrange(len(aug_batches_X)):
    s_batch_X, s_batch_Y = shuffle_batch(aug_batches_X[i], batches_Y[i])
    s_batches_X.append(s_batch_X)
    s_batches_Y.append(s_batch_Y)

print "Sample batch training images after shuffling"
# picgrid(s_batches_X[sample], s_batches_Y[sample])


def batch(batch_size):  # one shortcut function to execute all necessary functions to create a training batch
    batches_X, batches_Y = create_batches(batch_size, classes_num)

    aug_batches_X = []
    for batch in batches_X:
        aug_batch_X = augment_batch(batch)
        aug_batches_X.append(aug_batch_X)

    s_batches_X = []
    s_batches_Y = []

    for i in xrange(len(aug_batches_X)):
        s_batch_X, s_batch_Y = shuffle_batch(aug_batches_X[i], batches_Y[i])
        s_batches_X.append(s_batch_X)
        s_batches_Y.append(s_batch_Y)

    return s_batches_X, s_batches_Y