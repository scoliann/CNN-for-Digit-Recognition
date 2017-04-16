import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
tf.set_random_seed(0)

# Read in training data from train.csv
dfTrain = pd.read_csv('train.csv')
dfTrainFeatureVectors = dfTrain.drop(['label'], axis=1)
trainFeatureVectors = dfTrainFeatureVectors.values.astype(dtype=np.float32)
trainFeatureVectorsConvoFormat = trainFeatureVectors.reshape(42000, 28, 28, 1)

trainLabelsList = dfTrain['label'].tolist()
ohTrainLabelsTensor = tf.one_hot(trainLabelsList, depth=10)
ohTrainLabelsNdarray = tf.Session().run(ohTrainLabelsTensor).astype(dtype=np.float64)

# Read in testing data from test.csv
dfTest = pd.read_csv('test.csv')
testFeatureVectors = dfTest.values.astype(dtype=np.float32)
testFeatureVectorsConvoFormat = testFeatureVectors.reshape(28000, 28, 28, 1)

# Display an image read in from the CSV
# testFeatureVectorsConvoFormat values are: [2, 0, 9, 0, 3, 7, ...]
pixels = testFeatureVectorsConvoFormat[0].reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()

# Define Tensorflow graph
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])
lr = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)

K = 6  # first convolutional layer output depth
L = 12  # second convolutional layer output depth
M = 24  # third convolutional layer
N = 200  # fully connected layer (softmax)

W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1))  # 6x6 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))

# The model
stride = 1  # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2  # output is 14x14
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2  # output is 7x7
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
YY4 = tf.nn.dropout(Y4, pkeep)
Ylogits = tf.matmul(YY4, W5) + B5
Y = tf.nn.softmax(Ylogits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
predictions = tf.argmax(Y, 1)

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

def getBatch(i, size, trainFeatures, trainLabels):
    startIndex = (i * size) % 42000	# 42000 is the size of the train.csv data set
    endIndex = startIndex + size
    batch_X = trainFeatures[startIndex : endIndex]
    batch_Y = trainLabels[startIndex : endIndex]

    return batch_X, batch_Y


# You can call this function in a loop to train the model, 100 images at a time
def training_step(i):

    # training on batches of 100 images with 100 labels
    size = 100
    batch_X, batch_Y = getBatch(i, size, trainFeatureVectorsConvoFormat, ohTrainLabelsNdarray)

    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

    # compute training values
    if i % 20 == 0:

        '''
        When we sess.run here, we are calculating the accuracy and cross_entropy of the model on batch_X and batch_Y (ie. on 100 pieces of data)
        '''
        a, c = sess.run([accuracy, cross_entropy], {X: batch_X, Y_: batch_Y, pkeep: 1.0})
        print(str(i) + ": training accuracy:" + str(a) + " training loss: " + str(c) + " (lr:" + str(learning_rate) + ")")

    
    # compute test values
    if i % 100 == 0:
    
        '''
        When we sess.run here, we are calculating the accuracy and cross_entropy of the model on all of the data

	Note:  I am aware that displaying the accuracy on training data is fraudulent, as this is the accuracy on data on which we have trained.
	       If you would like a proper validation set of 10000, then change the 42000 in the getBatch() function to 32000.
        '''
        a, c = sess.run([accuracy, cross_entropy], {X: trainFeatureVectorsConvoFormat[-10000:], Y_: ohTrainLabelsNdarray[-10000:], pkeep: 1.0})
        print(str(i) + ": ********* test accuracy:" + str(a) + " test loss: " + str(c))
    

    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate, pkeep: 0.75})

# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
for i in range(10000+1): 
    training_step(i)

# Print the test accuracy on some data that was held out
# For this metric to mean anything, you must change the 42000 in the getBatch() function to 32000
a, c = sess.run([accuracy, cross_entropy], {X: trainFeatureVectorsConvoFormat[-10000:], Y_: ohTrainLabelsNdarray[-10000:], pkeep: 1.0})
print("\n ********* test accuracy:" + str(a) + " test loss: " + str(c))

# Get predictions on test data
p = sess.run([predictions], {X: testFeatureVectorsConvoFormat, pkeep: 1.0})

# Write predictions to csv file
results = pd.DataFrame({'ImageId': pd.Series(range(1, len(p[0]) + 1)), 'Label': pd.Series(p[0])})
results.to_csv('results.csv', index=False)












