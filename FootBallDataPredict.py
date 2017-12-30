# -*- coding: utf-8 -*-
"""
this is a tensorflow test
"""
import pandas as bPd
import numpy as bNp
import tensorflow as tf
import os
from numpy.random import RandomState
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

# =============================================================================
# ############### Super Params Define #####################
# =============================================================================
# Basic
g_tLayerDimension = [16 * 16 * 16, 1000, 100, 9, 3]
g_funActivateFunc = tf.nn.softmax
g_funLastActFunc  = tf.nn.softmax
g_nTotalSteps     = 8000000
g_nMiniBatchSize  = 500
g_nCheckRound     = 500
#Optimize
g_fOriLearnRate   = 0.7
g_nStep4RateDecay = 10000
g_fLearnDecayRate = 0.96
g_fKeepProb       = 0.50
g_bIsEnalbeBN     = False
g_fL2Regularation = 0.05
#Check
g_fExpectAccruacy = 0.1 + 0.1

# define Custmize Data
# Create DataSet: define column name
g_Column_name = ['Team1Goal', 'Team1Los', 'Team1Yellow',
                 'Team1Red', 'Team1WinRate', 'Team1PinRate', 'Team1LosRate', 'Team2Goal', 'Team2Los', 'Team1MatchCount',
                 'Team2Yellows', 'Team2Reds', 'Team2WinRate', 'Team2PinRate', 'Team2LosRate', 'Team2MatchCount', '3', '1', '0']

# define Continues!
g_bIsContinuesFromLast = False
g_strModelDir = "Model_FootBall"

# =============================================================================
# ############### Define Functions #####################
# =============================================================================
def get_cov2d(inputTensor, filter):
    return tf.nn.conv2d(
        input   = inputTensor,
        filter  = filter,
        strides = [1, 1, 1, 1],
        padding = 'SAME')
    
def get_max_pool2x2(inputTensor):
    return tf.nn.max_pool(
        value   = inputTensor,
        ksize   = [1, 2, 2, 1],
        strides = [1, 2, 2, 1],
        padding = 'SAME')

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
    name        = "weights", 
    shape       = shape,
    initializer = tf.truncated_normal_initializer(stddev = 1.0))

    if regularizer != None:
        tf.add_to_collection("losses", regularizer(weights))
    return weights

def get_biases_variable(shape):
    biases = tf.get_variable(
    name        = "biases", 
    shape       = shape,
    initializer = tf.constant_initializer(0.1))

    return biases

def get_batch_norm(inputTensor, curStep, isTestMode = False):
    offset     = tf.Variable(tf.zeros([inputTensor.get_shape()[1].value]))
    scale      = tf.Variable(tf.ones([inputTensor.get_shape()[1].value]))
    bnepsilon  = 1e-5

    mean, variance = tf.nn.moments(x = inputTensor, axes = [0])

    expMovingAvg   = tf.train.ExponentialMovingAverage(0.95, curStep)
    def mean_var_with_update():
        newMovingAvg = expMovingAvg.apply([mean, variance])
        with tf.control_dependencies([newMovingAvg]):
            return tf.identity(mean), tf.identity(variance)

    if isTestMode == True:
        mean = expMovingAvg.average(mean)
        variance = expMovingAvg.average(variance)
    else:
        mean, variance = mean_var_with_update()

    return tf.nn.batch_normalization(inputTensor, mean, variance, offset, scale, bnepsilon)

def build_net(dx, dy, nTrainDataSize, bIsNormed, nLearnRate):
    # Function Define
    def add_layer(tInputs, nIn_size, nOut_size, strLayer_name, nDataSize, fActivation_function=None, bIsNormed=False, isNeedDropOut = False):
        tWeights = tf.Variable(tf.truncated_normal([nIn_size, nOut_size], stddev=1.0))
        tBiases = tf.Variable(tf.zeros([1, nOut_size]) + 0.1)
        tWxplusb = tf.matmul(tInputs, tWeights) + tBiases
        if isNeedDropOut:
            tWxplusb = tf.nn.dropout(tWxplusb, keep_prob)
        tOutputs = tWxplusb

        # BatchNorm
        if bIsNormed:
            tOutputs    = get_batch_norm(tOutputs, nIterations, bIsTesing)

        if fActivation_function is not None:
            tOutputs = fActivation_function(tWxplusb)

        # regularization add Loss collection
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(g_fL2Regularation / nDataSize)(tOutputs))
        tf.summary.histogram(strLayer_name+'/outputs',tOutputs)
        return tOutputs

    if bIsNormed:
        dx = get_batch_norm(dx, nIterations, bIsTesing)

    x_firstTeam,x_secondTeam = tf.split(dx, [8, 8], 1)
    print(x_firstTeam.shape)
    print(x_secondTeam.shape)
    with tf.variable_scope('seperateData1'):
        w_sepd_1 = get_weight_variable([8, 256], None)
        b_sepd_1 = get_biases_variable([256])
        h_sepd_1 = tf.nn.softmax(tf.matmul(x_firstTeam, w_sepd_1) + b_sepd_1)

    with tf.variable_scope('seperateData2'):
        w_sepd_2 = get_weight_variable([8, 256], None)
        b_sepd_2 = get_biases_variable([256])
        h_sepd_2 = tf.nn.softmax(tf.matmul(x_secondTeam, w_sepd_2) + b_sepd_2)

    x_collect = tf.concat([h_sepd_1, h_sepd_2], 1)
    print(x_collect.shape)
    x_image = tf.reshape(x_collect, [-1, 16, 16, 2])
    with tf.variable_scope('conv2d1'):
        w_conv_1 = get_weight_variable([3, 3, 2, 16], None)
        b_conv_1 = get_biases_variable([16])
        h_conv_1 = tf.nn.relu(get_cov2d(x_image, w_conv_1) + b_conv_1)
        h_pool_1 = get_max_pool2x2(h_conv_1)
        h_pool_1_flat = tf.reshape(h_conv_1, [-1, 16 * 16 * 16]) # Prepare AllLinked Layer Shape

    #with tf.variable_scope('conv2d2'):
    #    w_conv_2 = get_weight_variable([3, 3, 16, 32], None)
    #    b_conv_2 = get_biases_variable([32])
    #    h_conv_2 = tf.nn.relu(get_cov2d(h_pool_1, w_conv_2) + b_conv_2)
    #    h_pool_2 = get_max_pool2x2(h_conv_2)

    #with tf.variable_scope('conv3d3'):
    #    w_conv_3 = get_weight_variable([3, 3, 32, 64], None)
    #    b_conv_3 = get_biases_variable([64])
    #    h_conv_3 = tf.nn.relu(get_cov2d(h_pool_2, w_conv_3) + b_conv_3)
    #    h_pool_3 = get_max_pool2x2(h_conv_3)

    # Apply Layers
    tCurLayer = h_pool_1_flat
    print(tCurLayer.shape)
    for i in range(1, len(g_tLayerDimension)):
        fActFunc = g_funActivateFunc
        isNeedDropOut = True
        if i == (len(g_tLayerDimension) - 1):
            fActFunc       = g_funLastActFunc
            isNeedDropOut = False
        tCurLayer = add_layer(tCurLayer, g_tLayerDimension[i - 1], g_tLayerDimension[i], 'hiddenLayer_%d' % i, nTrainDataSize, fActFunc, g_bIsEnalbeBN, isNeedDropOut)
        print(tCurLayer.shape)

    # define loss function
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(dy * tf.log(tf.clip_by_value(tCurLayer, 1e-10, 1.0)), 1))
    fLossFunc = cross_entropy + tf.add_n(tf.get_collection('losses'))
    tf.summary.scalar('loss', cross_entropy)

    # define verse deliver function
    train_step = tf.train.AdagradOptimizer(nLearnRate).minimize(fLossFunc, global_step = global_step)

    return [train_step, fLossFunc, tCurLayer]

# Read Data(From Web)
tData = bPd.read_csv('Data/FootData.csv')
# =============================================================================
# ############### Data Validation #####################
# =============================================================================
# Replace NAN
tData = tData.replace(to_replace='', value=bNp.nan)
# Drop NAN line
tData = tData.dropna(how='any')
# OutputData
tData.shape

#from sklearn.preprocessing import PolynomialFeatures
#pData = PolynomialFeatures().fit_transform(tData[g_Column_name[0:14]])

#print(pData.shape)
# =============================================================================
# ######### From Source Split Test Samples & Train Samples ############
# =============================================================================
x_train, x_test, y_train, y_test = train_test_split(tData[g_Column_name[0:16]],
                                                    tData[g_Column_name[16:19]],
                                                    test_size=0.25,
                                                    random_state=30)

# =============================================================================
# ############### Data Preproces #####################
# =============================================================================
#ss = StandardScaler()
#x_train = ss.fit_transform(x_train)
#x_test = ss.transform(x_test)

# =============================================================================
# ######### Form TensorFlow ############
# =============================================================================
# start a Session
dataset_size = len(y_train)
# define inputdata room
x = tf.placeholder(tf.float32, shape=(None, 16), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 3), name='y-input')
# define dropout
keep_prob = tf.placeholder(tf.float32)
# define curStep
nIterations = tf.placeholder(tf.int32)
bIsTesing   = tf.placeholder(tf.bool)

# define decay rate
global_step = tf.Variable(0, trainable = False)
nLearnRate  = tf.train.exponential_decay(g_fOriLearnRate, global_step, g_nStep4RateDecay, g_fLearnDecayRate, staircase = True)

# build net
train_step, cross_entropy, layer_graph = build_net(x, y_, dataset_size, g_bIsEnalbeBN, nLearnRate)


with tf.Session() as sess:
    #merged = tf.summary.merge_all()  
    #train_write = tf.summary.FileWriter("logs/train",sess.graph)
    #test_write = tf.summary.FileWriter("logs/test",sess.graph)

    tf.global_variables_initializer().run()

    if g_bIsContinuesFromLast == True:
        saver = tf.train.Saver()
        saver.restore(sess, "/Model_FootBall/footballmodel")

    for i in range(g_nTotalSteps):
        start = (i * g_nMiniBatchSize) % dataset_size
        end = min(start + g_nMiniBatchSize, dataset_size)

        # start to train
        sess.run(train_step, feed_dict={x: x_train[start:end], y_: y_train[start:end], keep_prob: g_nCheckRound, bIsTesing: False, nIterations: i})

        # Check&Save Model
        if i % g_nCheckRound == 0:
            # Calc cross_entropy
            total_cross_entropy = sess.run(
                cross_entropy, feed_dict={x: x_train, y_: y_train, keep_prob: 1.0, bIsTesing: False, nIterations: i})
            print("After %d training steps(s) cross entropy on all data is %g" % (i, total_cross_entropy))

            # accuracy by train_data
            correct_prediction = tf.equal(tf.argmax(layer_graph, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            ac = accuracy.eval({x: x_train, y_: y_train, keep_prob: 1.0, bIsTesing: True, nIterations: i})
            print(ac)

            # accuracy by test_data
            correct_prediction_test = tf.equal(tf.argmax(layer_graph, 1), tf.argmax(y_, 1))
            accuracy_test = tf.reduce_mean(tf.cast(correct_prediction_test, tf.float32))
            bc = accuracy_test.eval({x: x_test, y_: y_test, keep_prob: 1.0, bIsTesing: True, nIterations: i})
            print(bc)

            # form visual data: Need Tensor Board
            #train_result = sess.run(merged, feed_dict = {x:x_train, y_:y_train, keep_prob: 1.0})
            #test_result = sess.run(merged, feed_dict = {x:x_test, y_:y_test, keep_prob: 1.0})
            #train_write.add_summary(train_result, i)  
            #test_write.add_summary(test_result, i)

            # Save Model
            if (ac + bc) > g_fExpectAccruacy:
                f = open("out.txt", "a+")
                print("Details:")
                print("%r LearnRate: %f batch_size: %d ActFunc:softmax, keep_prob: %f" % (g_tLayerDimension, g_fOriLearnRate, g_nMiniBatchSize, g_fKeepProb), file = f)
                print("Cur Data Num %d, accuracy of train_data: %g, accuracy of test_data %g, cross_entry %g" % (i, ac, bc, total_cross_entropy), file = f)
                print("-------------------------------------------------\n")
                f.close()
                g_fExpectAccruacy = (ac + bc)

                # save models
                saver = tf.train.Saver()
                model_dir = g_strModelDir
                model_name = "footballmodel"
                if not os.path.exists(model_dir):
                    os.mkdir(model_dir)
                saver.save(sess, os.path.join(model_dir, model_name))
