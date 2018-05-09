# Construct model
from WideResNet2 import WideResNet
import tensorflow as tf

import h5py
import os
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3, 4"

from dataUtils2 import create_batches, getbatch
from dataUtils2 import augment_batch
from dataUtils2 import shuffle_batch


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

file = h5py.File('fashion_data.h5','r+')

#Retrieves all the preprocessed training and validation\testing data from a file

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

# classes_num = len(classes['label_names'])  # classes_num = no. of classes
classes_num = 10
batches_X, batches_Y = create_batches(X_train, Y_train, 64, classes_num)  # A demo of the function at work

# Since each batch will have almost equal no. of cases from each class, no batch should be biased towards some particular classes

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

#Hyper Parameters!

learning_rate = 0.01
init_lr = learning_rate
batch_size = 64
epochs = 500
layers = 16
beta = 0.0001 #l2 regularization scale
# = 1 #no. of models to be ensembled (minimum: 1)

K = 8 #(deepening factor)

n_classes = classes_num # another useless step that I made due to certain reasons.

# tf Graph input

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None,classes_num])
phase = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
model = WideResNet(x, keep_prob, phase, layers=layers, kval=K, scope='1')

# l2 regularization
weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='1regularize')

regularizer = 0
for i in xrange(len(weights)):
    regularizer += tf.nn.l2_loss(weights[i])

# cross entropy loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y) + beta * regularizer)

global_step = tf.Variable(0, trainable=False)

# optimizer
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                       momentum=0.9,
                                       use_nesterov=True).minimize(cost, global_step=global_step)

# Evaluate model
correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
prediction = tf.nn.softmax(logits=model)

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:  # Start Tensorflow Session
    saver = tf.train.Saver()  # Prepares variable for saving the model
    sess.run(init)  # initialize all variables
    step = 1
    loss_list = []
    acc_list = []
    val_loss_list = []
    val_acc_list = []
    best_val_acc = 0
    total_loss = 0
    total_acc = 0
    avg_loss = 0
    avg_acc = 0
    val_batch_size = batch_size
    threshold = 0.5  # if training accuracy is 100-threshold or less, training will stop

    while step <= epochs:

        # A little bit of Learning rate scheduling
        if step == 60:
            learning_rate = 0.01
        elif step == 120:
            learning_rate = 0.004
        elif step == 160:
            learning_rate = 0.0008

        batches_X, batches_Y = getbatch(X_train, Y_train, batch_size, n_classes)

        for i in xrange(len(batches_X)):
            # Run optimization operation (backpropagation)
            _, loss, acc = sess.run([optimizer, cost, accuracy],
                                    feed_dict={x: batches_X[i], y: batches_Y[i],
                                               keep_prob: 0.7,
                                               phase: True})
            total_loss += loss
            total_acc += acc

            if i % 100 == 0:
                print "Iter " + str((step - 1) * len(batches_X) + i + 1) + ", Minibatch Loss= " + \
                      "{:.3f}".format(loss) + ", Minibatch Accuracy= " + \
                      "{:.3f}%".format(acc * 100)

        total_val_loss = 0
        total_val_acc = 0
        val_loss = 0
        val_acc = 0
        avg_val_loss = 0
        avg_val_acc = 0

        i = 0
        count = 0
        while i < len(X_val):

            if i + val_batch_size < len(X_val):
                val_loss, val_acc = sess.run([cost, accuracy],
                                             feed_dict={x: X_val[i:i + val_batch_size],
                                                        y: Y_val[i:i + val_batch_size],
                                                        keep_prob: 1,
                                                        phase: False})
            else:
                val_loss, val_acc = sess.run([cost, accuracy],
                                             feed_dict={x: X_val[i:],
                                                        y: Y_val[i:],
                                                        keep_prob: 1,
                                                        phase: False})

            total_val_loss = total_val_loss + val_loss
            total_val_acc = total_val_acc + val_acc
            count += 1

            i += val_batch_size

        avg_val_loss = total_val_loss / count  # Average validation loss
        avg_val_acc = total_val_acc / count  # Average validation accuracy

        val_loss_list.append(avg_val_loss)  # Storing values in list for plotting later on.
        val_acc_list.append(avg_val_acc)  # Storing values in list for plotting later on.

        avg_loss = total_loss / len(batches_X)  # Average mini-batch training loss
        avg_acc = total_acc / len(batches_X)  # Average mini-batch training accuracy
        loss_list.append(avg_loss)  # Storing values in list for plotting later on.
        acc_list.append(avg_acc)  # Storing values in list for plotting later on.

        total_loss = 0
        total_acc = 0

        print "\nEpoch " + str(step) + ", Validation Loss= " + \
              "{:.3f}".format(avg_val_loss) + ", validation Accuracy= " + \
              "{:.3f}%".format(avg_val_acc * 100) + ""
        print "Epoch " + str(step) + ", Average Training Loss= " + \
              "{:.3f}".format(avg_loss) + ", Average Training Accuracy= " + \
              "{:.3f}%".format(avg_acc * 100) + ""

        if avg_val_acc >= best_val_acc:  # When better accuracy is received than previous best validation accuracy

            best_val_acc = avg_val_acc  # update value of best validation accuracy received yet.
            saver.save(sess, 'Model_Backup/model.ckpt')  # save_model including model variables (weights, biases etc.)
            print "Checkpoint created!"

        print ""

        if (100 - (avg_acc * 100)) <= threshold:
            print "\nConvergence Threshold Reached!"
            break

        step += 1

    print "\nOptimization Finished!\n"

    print "Best Validation Accuracy: %.3f%%" % ((best_val_acc) * 100)

    print 'Loading pre-trained weights for the model...'
    saver = tf.train.Saver()
    saver.restore(sess, 'Model_Backup/model.ckpt')
    sess.run(tf.global_variables())
    print '\nRESTORATION COMPLETE\n'

    print 'Testing Model Performance...'
    test_batch_size = batch_size
    total_test_loss = 0
    total_test_acc = 0
    test_loss = 0
    test_acc = 0
    avg_test_loss = 0
    avg_test_acc = 0

    i = 0
    count = 0
    while i < len(X_test):

        if (i + test_batch_size) < len(X_test):
            test_loss, test_acc = sess.run([cost, accuracy],
                                           feed_dict={x: X_test[i:i + test_batch_size],
                                                      y: Y_test[i:i + test_batch_size],
                                                      keep_prob: 1,
                                                      phase: False})
        else:
            test_loss, test_acc = sess.run([cost, accuracy],
                                           feed_dict={x: X_test[i:],
                                                      y: Y_test[i:],
                                                      keep_prob: 1,
                                                      phase: False})

        total_test_loss = total_test_loss + test_loss
        total_test_acc = total_test_acc + test_acc
        count += 1

        i += test_batch_size

    avg_test_loss = total_test_loss / count  # Average test loss
    avg_test_acc = total_test_acc / count  # Average test accuracy

    print "Test Loss = " + \
          "{:.3f}".format(avg_test_loss) + ", Test Accuracy = " + \
          "{:.3f}%".format(avg_test_acc * 100)