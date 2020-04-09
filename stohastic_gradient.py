
import numpy as np
import tensorflow.compat.v2 as tf 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


class Model(object):
	""" class for implementation of stohastic gradient descent 
	"""
	def __init__(self):
		self.W = None
		self.b = None

	def initial_weights(self, input_size, output_size):
		if(self.W == None and self.b == None):
			self.W = tf.Variable(tf.random.normal(mean = 0.0, stddev = 0.01,
				shape = (input_size, output_size)), dtype = tf.float32)
			self.b = tf.Variable(tf.zeros(shape = (1, output_size)), dtype = tf.float32)
		if(input_size == 1 and output_size == 1):
			self.W = tf.Variable(self.W.numpy()[0][0], dtype = tf.float32)
			self.b = tf.Variable(self.b.numpy()[0][0], dtype = tf.float32)

	def __str__(self):
		return "W: {!s}, \n b: {!s}".format(self.W.numpy(),
										 self.b.numpy())

	def __call__(self, X): # X is tensor object should be
			if(self.W.shape.__len__() == 0 and self.b.shape.__len__() == 0):
				return self.W * X + self.b
			else: ## matrix
				return tf.matmul(X, self.W) + self.b


def loss_(y, p):
	return tf.reduce_mean(tf.square(y - p)) / y.shape[0]

def to_tf(tensor):
	return tf.convert_to_tensor(tensor, dtype = tf.float32)

def train(model, inputs, outputs, lr = 0.01):
	with tf.GradientTape(persistent = True) as g:
		g.watch((model.W, model.b, inputs))
		current_loss = loss_(outputs, model(inputs))
		dW, db = g.gradient(current_loss, [model.W, model.b])
		model.W.assign_sub(dW * lr)
		model.b.assign_sub(db * lr)


class test_mod(object):

	def __init__(self):
		self.W = None
		self.b = None

	def init_weights(self, init_W, init_b):
		self.W = tf.Variable(init_W, dtype = tf.float32)
		self.b = tf.Variable(init_b, dtype = tf.float32)

	def __call__(self, x):
		return self.W * x + self.b


def test_(loss):
	pass


class data_1_d(object):

	def __init__(self, initial_weights = [2., 3.]):

		self.W_1d = tf.constant(initial_weights[0], dtype = tf.float32)
		self.b_1d = tf.constant(initial_weights[1], dtype = tf.float32)

		self.X_data = None 
		self.y_data = None

	def _initial_data(self, std = 1, mean = 0.0, num_data = 1000):
		if(all(self.X_data == None) and all(self.y_data == None)):
			self.X_data = np.linspace(-2, 2, num_data)
			self.y_data = self.W_1d.numpy() * self.X_data + self.b_1d.numpy() + np.random.normal(scale = std, loc = mean, 
																		size = self.X_data.shape)
			self.X_data = self.X_data[:,np.newaxis]
			self.y_data = self.y_data[:,np.newaxis]
            
	def plot_1_d(self):
		self._initial_data()
		plt.figure(figsize = (6, 5))
		print(type(self.X_data), type(self.y_data))
		plt.scatter(self.X_data, self.y_data)
		plt.grid()
		plt.legend(['data'])
		plt.show()   

	def _split(self):
		if(all(self.X_data != None) and all(self.y_data != None)):
			X_train, X_test, y_train, y_test = train_test_split(self.X_data, self.y_data, train_size = 0.7)
			return [to_tf(X_train), to_tf(X_test), to_tf(y_train), to_tf(y_test)]      
        
class Datasets(object):

	def __init__(self, n_samples = 100, n_features = 2, n_targets = 3, noise = 10, random_state = 1): # numpy objects
		self.X, self.y = datasets.make_regression(n_samples = n_samples, n_features = n_features, 
	           n_targets = n_targets, noise = noise, random_state = random_state)
		if(n_targets == 1):
			self.y = self.y[:, np.newaxis]
		self.X = to_tf(self.X)
		self.y = to_tf(self.y)
		self.data = None

	def split_data(self, train_size = 0.7): # tensor obj
		[X_train, X_test, y_train, y_test] = train_test_split(self.X.numpy(), self.y.numpy(), train_size = 0.7)
		if(self.data == None):
			self.data = [X_train, X_test, y_train, y_test] 
		return [to_tf(X_train), to_tf(X_test), to_tf(y_train), to_tf(y_test)]

	def plot(self, model = None): # only for one dimensional tasks
		if(self.data == None):
			print('Should call after split_data!')
		else:
			[X_train, X_test, y_train, y_test] = self.data
			plt.figure(figsize = (6,5))
			plt.scatter(X_train, y_train, c = 'r')
			plt.scatter(X_test, y_test, c = 'b')
			if(model != None):
				x_data = np.linspace(np.min(self.X.numpy()), np.max(self.X.numpy()))
				predict_ = [model.W.numpy() * x + model.b.numpy() for x in x_data]
				plt.plot(x_data, predict_)
				plt.plot()
			
			plt.legend(['Train, Test'])
			plt.grid()
			plt.show()

data_obj = Datasets(n_samples = 1000, n_targets = 3, n_features = 2, noise = 5)
X_train, X_test, y_train, y_test = data_obj.split_data(train_size = 0.7)

#data_obj.plot()

model = Model()
model.initial_weights(X_train.shape[1], y_train.shape[1])


def GD(model, inputs, outputs, epochs = 100):

	Ws, bs = [], []
	Loss_ = []
	for epoch in range(epochs):
		Ws.append(model.W)
		bs.append(model.b)
		Loss_.append(loss_(outputs, model(inputs)))

		train(model, inputs, outputs, lr = 0.1)
		#print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %
		#			(epoch, Ws[-1], bs[-1], current_loss))

	plt.plot(range(epochs), Loss_, 'r')
	plt.legend(['Loss'])
	plt.grid()
	plt.show()
	return Loss_

def plot_with_learned(model, data_obj): # only for onedimensional task
	data_obj.plot()
	X_train = data_obj.data[0] ## here not a tensor
	x_data = np.linspace(np.min(data_obj.X.numpy()), np.max(data_obj.X.numpy()))    
	predict_ = [model.W.numpy() * x + model.b.numpy() for x in x_data]
	plt.plot(x_data, predict_)
	plt.show()

Loss_ = GD(model, X_train, y_train, epochs = 1000)
Loss_[-1]
#data_obj.plot(model)
model.W, model.b, Loss_[-1]


class test_model(object):

	def __init__(self):
		self.W = None
		self.b = None

	def init_weights(self, init_W, init_b):
		self.W = tf.Variable(init_W, dtype = tf.float32)
		self.b = tf.Variable(init_b, dtype = tf.float32)

	def __call__(self, x):
		return self.W * x + self.b

	def __str__(self):
		return "W: {!s}, b: {!s}".format(self.W.numpy(), self.b.numpy())



def test_(loss):
	pass


class data_1_d(object):
 
	
    
	def __init__(self, initial_weights = [1, 3.]):

		self.W_1d = tf.constant(initial_weights[0], dtype = tf.float32)
		self.b_1d = tf.constant(initial_weights[1], dtype = tf.float32)

		self.X_data = None 
		self.y_data = None
		self.flag_init = False

	def _initial_data(self, std = 1, mean = 0.0, num_data = 1000):
		if(self.flag_init == False):
			self.flag_init = True
			self.X_data = np.random.normal(size = (num_data, ))
			self.y_data = self.W_1d.numpy() * self.X_data + self.b_1d.numpy() + np.random.normal(scale = std, loc = mean, 
																		size = self.X_data.shape)
			self.X_data = self.X_data[:,np.newaxis]
			self.y_data = self.y_data[:,np.newaxis]
            
	def plot_1_d(self):
		self._initial_data()
		plt.figure(figsize = (6, 5))
		print(type(self.X_data), type(self.y_data))
		plt.scatter(self.X_data, self.y_data)
		plt.grid()
		plt.legend(['data'])
  

	def _split(self):
		if(all(self.X_data != None) and all(self.y_data != None)):
			X_train, X_test, y_train, y_test = train_test_split(self.X_data, self.y_data, train_size = 0.7)
			return [to_tf(X_train), to_tf(X_test), to_tf(y_train), to_tf(y_test)]      

    
def train(model, inputs, outputs, lr = 0.1):
	with tf.GradientTape(persistent = True) as g:
		g.watch((model.W, model.b, inputs))
		current_loss = loss_(outputs, model(inputs))
		dW, db = g.gradient(current_loss, [model.W, model.b])
		model.W.assign_sub(dW * lr)
		model.b.assign_sub(db * lr)


def train_numpy(model, inputs, outputs, lr = 0.1):
	grad_W = (2 / inputs.shape[0])*np.dot(inputs.numpy().T, 
		          (model(inputs) - outputs).numpy())
	grad_b = (2 / inputs.shape[0]) * ((model(inputs) - outputs).numpy())
	grad_b = np.mean(grad_b, axis = 0)
	model.W.assign_sub(lr * grad_W[0][0])
	model.b.assign_sub(lr * grad_b[0])
    
    
test_M = test_mod()
test_M.init_weights(init_W = 2., init_b = 1.)

data = data_1_d()
data._initial_data()
data_ = data._split()
data.plot_1_d()

inputs = data_[0]
outputs = data_[2]
print(test_M.W.numpy(), test_M.b.numpy())
train(test_M, inputs, outputs)
print(test_M.W.numpy(), test_M.b.numpy())
test_M(inputs).shape
loss_(test_M(inputs), outputs)

test_M(inputs).numpy()
plt.scatter(inputs.numpy(), test_M(inputs).numpy(), c ='r')

train_numpy(test_M, inputs, outputs)
print(test_M.W.numpy(), test_M.b.numpy())

Ws, bs = [], []
epochs = range(100)
for epoch in epochs:
	Ws.append(test_M.W.numpy())
	bs.append(test_M.b.numpy())
	current_loss = loss_(outputs, test_M(inputs))
    
	train_numpy(test_M, inputs, outputs, lr = 0.1)
    
print('hello',test_M.W.numpy(), test_M.b.numpy())
