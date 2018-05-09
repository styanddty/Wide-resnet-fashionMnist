import h5py

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

from dataUtils2 import getbatch

DATA_DIR2 = 'please input the path of fashion mnist dataset'

mnist = input_data.read_data_sets(DATA_DIR2, one_hot=False, validation_size=0)
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

print train_data.shape, train_labels.shape
test_data = mnist.test.images  # Returns np.array
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

print test_data.shape, test_labels.shape

# for i in range(10):
#     print test_labels[i]
# print type(train_labels[0])
classes_num = 10
train_labels2 = np.zeros((len(train_data), classes_num), np.float32)
test_labels2 = np.zeros((len(test_data), classes_num), np.float32)

for i in range(len(train_labels)):
    index = train_labels[i]
    train_labels2[i][index] = 1

for i in range(len(test_labels)):
    index = test_labels[i]
    test_labels2[i][index] = 1
train_labels = train_labels2
test_labels = test_labels2

train_data2 = np.zeros((len(train_data), 28, 28, 1), dtype=np.float32)
for i in xrange(len(train_data)):
    train_data2[i] = np.reshape(train_data[i], (28, 28, 1))

test_data2 = np.zeros((len(test_data), 28, 28, 1), dtype=np.float32)
for i in xrange(len(test_data)):
    test_data2[i] = np.reshape(test_data[i], (28, 28, 1))
train_data = train_data2
test_data = test_data2


def mean_std_normalization(X, mean, calculate_mean=True):
    channel_size = X.shape[3]
    for i in xrange(channel_size):
        if calculate_mean == True:
            mean[i] = np.mean(X[:,:,:,i])
        variance = np.mean(np.square(X[:,:,:,i]-mean[i]))
        deviation = np.sqrt(variance)
        X[:,:,:,i] = (X[:,:,:,i]-mean[i])/deviation
    return X
channel_size = train_data.shape[3]
mean = np.zeros((channel_size),dtype=np.float32)
train_data = mean_std_normalization(train_data, mean)
test_data = mean_std_normalization(test_data, mean)

import h5py
file = h5py.File('fashion_data.h5','w')
file.create_dataset('train_data', data=train_data)
file.create_dataset('train_labels', data=train_labels)

file.create_dataset('test_data', data=test_data)
file.create_dataset('test_labels', data=test_labels)

file.close()
print "say hello"
print train_data[0].shape, train_labels[0].shape
print test_data[0].shape, test_labels[0].shape
print train_data.shape, train_labels.shape
print test_data.shape, test_labels.shape

file = h5py.File('fashion_data.h5','r+')
X_train = file['train_data'][...]
Y_train = file['train_labels'][...]

X_val = file['test_data'][...]
Y_val = file['test_labels'][...]

X_test = file['test_data'][...]
Y_test = file['test_labels'][...]

# Unpickles and retrieves class names and other meta informations of the database
# classes = unpickle('cifar-10-batches-py/batches.meta') #keyword for label = label_names

print("Training sample shapes (input and output): "+str(X_train.shape)+" "+str(Y_train.shape))
print("Validation sample shapes (input and output): "+str(X_val.shape)+" "+str(Y_val.shape))
print("Testing sample shapes (input and output): "+str(X_test.shape)+" "+str(Y_test.shape))
# batch_size = 64
# n_classes = 10
#
# batches_X, batches_Y = getbatch(X_train, Y_train, batch_size, n_classes)
# print len(batches_X), batches_X[0].shape
# print len(batches_Y), batches_Y[0].shape
import matplotlib.pyplot as plt
from scipy.misc import toimage
plt.figure(figsize=(7,7))
ax=[]
for i in xrange(0, 25):
    img = toimage(np.reshape(X_train[i], (-1, 28)))
    ax.append(plt.subplot(5,5,i+1))
    ax[i].set_title(np.argmax(Y_train[i]), y=-0.3)
    ax[i].set_axis_off()
    plt.imshow(img)
plt.subplots_adjust(hspace=0.3)
plt.axis('off')
# plt.show()
plt.savefig("dataDemoFashion3")