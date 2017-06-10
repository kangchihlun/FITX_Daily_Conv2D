# coding: utf-8
import tensorflow as tf
import numpy as np
from pylab import plt
import csv
import os
import pandas as pd
import random

# CHANGE THIS , training data root path
DataPath_FITX_day = "D:\\00Trunk\\02PY\\MachineLearning\\00_tools\\FITX_DataTools\\taifex_data_process\\taifex_data_process\\output_data\\"

# CHANGE THIS , num data for test(predict)
num_test_files = 50

# CHANGE THIS , percent of data to do validation
validation_percentage = 0.15

# CHANGE THIS , each batch take out each time
batchSize = 10

# CHANGE THIS , num epoch of training each time
numEpoch = 1000

'''
        DATA    PREPARATION
'''

def listdir_joined(path):
    return [os.path.join(path, entry) for entry in os.listdir(path)]

# path = DataPath_FITX_train + "\\train.txt" outArr=[]
def getfileRecursive(path,outArr,fileType='csv'):
    if(os.path.isfile(path)):
        _type_ = (path.split("."))[-1]
        if(fileType == _type_):
            outArr.append(path)
    elif(os.path.isdir(path)):
        _folders_ = [x for x in listdir_joined(path)]
        for f in _folders_ :
            getfileRecursive(f,outArr,fileType)

def readFilesIntoNumPyArr(filePaths,onehot=True,requiredShpSize=784): # filePaths = [all_data_sets_fullName[1]]
    _Batches = []
    for tra in filePaths: # tra = filePaths[0]
        numArr = [] # len(numArr)
        f = open(tra,'r')  
        lines = f.readlines() #len(lines)
        for line in lines: # line = lines[0]
            print(line)
            line_spl = line.split(',') # len(line_spl)
            onehotarr = []
            for a in line_spl: # a = line_spl[1]
                if(not '\n' in a):
                    parsed = 0.0
                    try:parsed = float(a)
                    except:pass
                    
                    if(onehot):
                        onehotarr.append(parsed)
                    else:
                        numArr.append(parsed)
            if(onehot):
                _Batches.append(onehotarr)
        f.close()
        if(not onehot):
            while(len(numArr) < requiredShpSize):
                numArr.append(0)
            _Batches.append(numArr)
    return np.array(_Batches) # len(TrainBatches)




# all csv data
all_data_sets_fullName = []
getfileRecursive(DataPath_FITX_day,all_data_sets_fullName)

#for ff in all_data_sets_fullName:
    #print (ff)

# EXAMING OF MNIST DATA STRUCTURE 
# print (ss)
# JUST TO SHOW MNIST DATA STRUCTURE <NDArray<NDArray<float32>>>
# import tensorflow.examples.tutorials.mnist.input_data as input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# batch_xs, batch_ys = mnist.validation.images, mnist.validation.labels
# batch_xs, batch_ys = mnist.train.next_batch(55000)  1100/55000=0.02
# type(batch_xs)
# len(batch_xs)
# type(batch_xs[0])
# batch_xs[0][-1]
# len(batch_ys)
# batch_ys[0]


# #########################################################
# Split all data into Training Files and Test Files(Random)
# #########################################################

indxCol_forTest=[]
while(len(indxCol_forTest) < num_test_files):
    nexRandVal = random.randrange(len(all_data_sets_fullName))
    if(not nexRandVal in indxCol_forTest):
        indxCol_forTest.append(nexRandVal)
# len(indxCol_forTest)

# # Read All Labels into NumPy Array
train_labels=[]
test_labels=[]
all_Labels = readFilesIntoNumPyArr([all_data_sets_fullName[-1]],onehot=True)
# len(all_Labels)

# Add Test Label
for idx in indxCol_forTest:
    test_labels.append(all_Labels[idx])
# type(test_labels[0])
# Add Train Label
for lbIdx in range(0,len(all_Labels)):
    if(not lbIdx in indxCol_forTest):
        train_labels.append(all_Labels[lbIdx])
# len(train_labels) + len(test_labels) == len(all_Labels)
# train_labels[0]


#
train_batches=[]
test_batches=[]
all_batches = readFilesIntoNumPyArr(all_data_sets_fullName[:-2],onehot=False)
# Add Train dataset
for idx in indxCol_forTest:
    test_batches.append(all_batches[idx]) # idx = indxCol_forTest[2]
    
for lbIdx in range(0,len(all_batches)):
    if(not lbIdx in indxCol_forTest):
        train_batches.append(all_batches[lbIdx])

# ############################################
# extract validation from train train_batches
# ############################################
validationQty = int(len(train_batches) * validation_percentage)

# validation Index
validationIndexCol = []
while(len(validationIndexCol) < validationQty):
    nexRandVal = random.randrange(len(train_batches))
    if(not nexRandVal in validationIndexCol):
        validationIndexCol.append(nexRandVal)

data_tr = []
data_tr_label = []
data_validation = []
data_validation_label = []
for idx in validationIndexCol:
    data_validation.append(train_batches[idx]) # idx = indxCol_forTest[2]
    data_validation_label.append(train_labels[idx])    
    
for lbIdx in range(0,len(train_batches)): # lbIdx = 2
    if(not lbIdx in validationIndexCol):
        data_tr.append(train_batches[lbIdx])
        data_tr_label.append(train_labels[lbIdx]) 

# ratioOfBatchToTrain = batchSize / len(data_tr)
# ValidationBatchSize = int(len(data_validation_label) * ratioOfBatchToTrain)
# 

# len(train_batches) + len(test_batches) == len(all_batches)
# train_batches[0]
# exam data corrections here using your own eyes
# test_batches[3] -- No.153 
# test_labels[3]
# len(train_batches[2])
# len(all_batches[2])

xinput_size = len(train_batches[0])
yinput_size = len(train_labels[0])





'''
        TRAINING    PROCESS
'''

'''
def add_layer(inputs, size_in, size_out, layer_name, activation=None):
    with tf.name_scope(layer_name) as scope:
        with tf.name_scope("weights"):
            W = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
            W_hist = tf.summary.histogram("%s/weights"%layer_name, W)
            
        with tf.name_scope("biases"):
            b = tf.Variable(tf.zeros([size_out]), name="bias")
            b_hist = tf.summary.histogram("%s/biases"%layer_name, b)
        with tf.name_scope("Wx_plus_b"):
            net = tf.matmul(inputs, W) + b
            net_hist = tf.summary.histogram("%s/net"%layer_name, net)
        if activation == None:
            outputs = net
        else:
            outputs = activation(net)
        outputs_hist = tf.summary.histogram("%s/outputs"%layer_name, outputs)
    return outputs



# add_layer(x_,     784, 200, "layer1", tf.nn.relu)
# add_layer(layer1, 200, 200, "layer2", tf.nn.relu)
# add_layer(layer2, 200, 10,  "layer3", tf.nn.softmax)

# - take a glance at tensorboard!
# # Multi-layer Neural Network
multi_nn = tf.Graph()
with multi_nn.as_default():
    with tf.name_scope("inputs") as scope:
        x_ = tf.placeholder(tf.float32, [None, xinput_size], name="x_input")
        y_ = tf.placeholder(tf.float32, [None, yinput_size], name="y_input")
    
    layer1 = add_layer(x_, xinput_size, 200, "layer1", tf.nn.relu)
    layer2 = add_layer(layer1, 200, 200, "layer2", tf.nn.relu)
    layer3 = add_layer(layer2, 200, 100, "layer3", tf.nn.relu)
    y = add_layer(layer3, 100, yinput_size, "layer3", tf.nn.softmax)
    
    with tf.name_scope("loss") as scope:
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y+1e-10), name="cross_entropy")
        ce_summ = tf.summary.scalar("cross_entropy", cross_entropy)
        
    optimizer = tf.train.GradientDescentOptimizer(0.01, name="gradient")
    trainer = optimizer.minimize(cross_entropy, name="trainer")

    with tf.name_scope("test") as scope:
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_summ = tf.summary.scalar("accuracy", accuracy)
            
#


with tf.Session(graph=multi_nn) as sess: # sess = tf.Session(graph=multi_nn)
    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()
    writer_train = tf.summary.FileWriter("logs/k_layer/train/", sess.graph)
    writer_valid = tf.summary.FileWriter("logs/k_layer/valid/", sess.graph)
    sess.run(init)
    
    # Save Load Last Time Training Result
    checkpoint = tf.train.get_checkpoint_state("DNN_Pyalgo_FITX_DAY")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
    
    i = 0
    while(i < len(data_tr)):
        #batch_xs, batch_ys = mnist.train.next_batch(100)
        # extract batch pieces each time
        batch_xs = []
        batch_ys = []
        for k in range(batchSize):
            batch_xs.append(data_tr[i+k])
            batch_ys.append(data_tr_label[i+k])
        
        train_summ, accu_train, _ = sess.run([merged, accuracy, trainer],feed_dict={x_:np.array(batch_xs), y_:np.array(batch_ys)})
        writer_train.add_summary(train_summ, i)
        
        # Validation  Error
        if i%50 == 0: # len(batch_xs[1])
            valid_summ, accu_valid = sess.run([merged, accuracy],feed_dict={x_:np.array(data_validation), y_:np.array(data_validation)})
            writer_valid.add_summary(valid_summ, i)
            print ("iter:%s, train:%s, valid:%s"%(i, accu_train, accu_valid))
        
        i += batchQty
    
    #batch_xs, batch_ys = mnist.test.images, mnist.test.labels
    accu_test = sess.run(accuracy, feed_dict={x_:np.array(test_batches), y_:np.array(test_labels)})
    print ("-----------")
    print ("test:%s"%accu_test)

'''





#
#
'''
        Deep Mnist Conv2D Official
'''
#
#

from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, xinput_size])
y_ = tf.placeholder(tf.float32, shape=[None, yinput_size])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

#import math
#res = math.sqrt(784)

W_conv1 = weight_variable([5, 5, 1, 32]) 
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, yinput_size])
b_fc2 = bias_variable([yinput_size])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
 
 
# Save Load Last Time Training Result
checkpoint = tf.train.get_checkpoint_state("DNN_Pyalgo_FITX_DAY")
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("Successfully loaded:", checkpoint.model_checkpoint_path)
else:
    print("Could not find old network weights")


i = 0
while(i < len(train_batches)):
    # batch = mnist.train.next_batch(50) # len(batch[0][0])
    # extract batch pieces each time
    batch_xs = []
    batch_ys = []
    numExtract = batchSize if (i+batchSize < len(train_batches)) else len(train_batches)-i
    #print ("numExtract" + str(numExtract))
    k = 0
    for k in range(numExtract):
        batch_xs.append(train_batches[i+k])
        batch_ys.append(train_labels[i+k])
    
    train_step.run(feed_dict={x:batch_xs , y_: batch_ys, keep_prob: 0.5})
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_:batch_ys, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        
    i += batchSize


print("test accuracy %g"%accuracy.eval(feed_dict={x:np.array(test_batches), y_: np.array(test_labels), keep_prob: 1.0}))