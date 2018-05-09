import numpy as np
import random


def create_batches(X_train, Y_train, batch_size, classes_num=10):
    X_train_F = []
    for i in xrange(0, classes_num):
        X_train_F.append([])
    for i in xrange(0, len(X_train)):
        l = np.argmax(Y_train[i])  # l for label (in this case it's basically the index of class value elemenmts)
        # (Y_train is one hot encoded. Argmax returns the index for maximum value which should be 1 and
        # that index should indicate the value)
        X_train_F[l].append(X_train[i])

    smoothing_factor = 0.1  # for label smoothing
    s = int(batch_size / classes_num)  # s denotes samples taken from each class to create the batch.
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


def random_crop(img):
    t = 32
    # result = np.zeros_like((img))
    c = np.random.randint(0, 5)
    if c == 0:
        crop = img[4:t, 0:-4]
    elif c == 1:
        crop = img[0:-4, 0:-4]
    elif c == 2:
        crop = img[2:-2, 2:-2]
    elif c == 3:
        crop = img[4:t, 4:t]
    elif c == 4:
        crop = img[0:-4, 4:t]

    # translating cropped position
    # over the original image
    c = np.random.randint(0, 5)
    if c == 0:
        img[4:t, 0:-4] = crop[:]
    elif c == 1:
        img[0:-4, 0:-4] = crop[:]
    elif c == 2:
        img[2:-2, 2:-2] = crop[:]
    elif c == 3:
        img[4:t, 4:t] = crop[:]
    elif c == 4:
        img[0:-4, 4:t] = crop[:]

    return img


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


def getbatch(X_train, Y_train, batch_size, classes_num = 10):  # one shortcut function to execute all necessary functions to create a training batch

    batches_X, batches_Y = create_batches(X_train, Y_train, batch_size, classes_num)
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
