from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
rng = np.random

# Parameters.
learning_rate = 0.15
training_steps = 2000
display_step = 10

X = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
              7.042,10.791,5.313,7.997,5.654,9.27,3.1])
Y = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
              2.827,3.465,1.65,2.904,2.42,2.94,1.3])
#n_samples = X.shape[0]
#################### 1dimensional ###################
n_samples = 1000

True_W = 1.
True_b = 2.

X = np.random.normal(loc = 0. ,size = (n_samples,))
Y = X * True_W + True_b + np.random.normal(loc = 0. , size = (n_samples, ), scale = 0.3)

X_ = X[:, np.newaxis]
Y_ = Y[:, np.newaxis]

W = tf.Variable(np.random.normal(size = (1, 1)), name="weight")
b = tf.Variable(np.random.normal(size  = (1, 1)), name="bias")
print("initial_ weight:", W.numpy(), b.numpy(), '\n')
#X = X_
#Y = Y_
##################### end ###########################

def data_(n_features = 2, n_targets = 3, n_sapmles = 100, noise = 10, random_state = 42):
    X, Y = datasets.make_regression(n_features = n_features, n_targets = 3, n_samples = 100,
                                   noise=noise, random_state = random_state)
    [X_train, X_test, y_train, y_test] = train_test_split(X, Y, train_size = 0.7)
    data = [X_train, X_test, y_train, y_test]
    return data
###create data and create a new W_, and b_ ###
###############
data = data_(n_sapmles = 1000)
X_train = data[0]
y_train = data[2]
W_ = tf.Variable(tf.random.normal(mean = 0.0, stddev = 0.01,
                    shape = (X_train.shape[1],  y_train.shape[1])), dtype = tf.float32)

b_ = tf.Variable(tf.zeros(shape = (1, y_train.shape[1])), dtype = tf.float32)

W = W_
b = b_
X = X_train
Y = y_train
##############

# Linear regression (Wx + b).
def linear_regression(x):
    return W * x + b
########
def linear_regression_Mat(x):
    ### нужно из типа numpy где double был и где tensor ###
    if(x.shape.__len__() == 2):
        x_tf = tf.convert_to_tensor(x, dtype = tf.float32) # возможно для одномерия стоит это убрать...?
    return tf.matmul(x_tf, W) + b
########
linear_regression = linear_regression_Mat
########


# Mean square error.
def mean_square(y_pred, y_true):
    return tf.reduce_sum(tf.pow(y_pred - y_true, 2)) / (2 * n_samples)

# Stochastic Gradient Descent Optimizer.
#optimizer = tf.optimizers.SGD(learning_rate)

def run_optimization():
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)

        # Compute gradients.
        gradients = g.gradient(loss, [W, b])
        #print("grads shape: ", gradients)
        #print(gradients[0].shape, gradients[1].shape)
        W.assign_sub(learning_rate * gradients[0])
        b.assign_sub(learning_rate * gradients[1])
    
    
    # Update W and b following gradients.
#    optimizer.apply_gradients(zip(gradients, [W, b]))


def plot_total(X, Y):
    global linear_regression
    print(X.shape, Y.shape)
    if(X.shape.__len__() == 1):
        plt.plot(X, Y, 'ro', label = 'Original data')
        plt.plot(X, linear_regression(X).numpy()[:, 0], label = 'Fitted line')
        plt.legend()
        plt.show()


print("before learning: ")
plot_total(X, Y)
for step in range(1, training_steps + 1):
    # Run the optimization to update W and b values.
    run_optimization()
    
    if step % display_step == 0:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)
        if(X.shape.__len__() != 2):
            print("step: {!s} loss: {!s}, W: {!s}, b: {!s}".format(step, loss, W.numpy(), b.numpy()))
        elif(X.shape.__len__() == 2):
            print("step: {!s} loss: {!s}".format(step, loss))
        
plot_total(X, Y)

#X_, Y_

# Можно подумать как менять learning rate в зависимости от того, какой ритм идёт то есть градиенты очень малы