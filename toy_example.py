

try:
    import tensorflow.compat.v1 as tf
except Exception:
    pass

#import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dim_ = 2
def create_tf_fill_value(shape, value):
    assert isinstance(shape, tuple)
    data_x_tf = np.asarray(np.zeros((shape)) + value, np.float32)
    data_x_tf = tf.convert_to_tensor(data_x_tf, np.float32)
    return data_x_tf
    
def show_tf(tensor_):
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(tensor_))
def fill_value_vector(value, shape):
    tensor_fill_value = tf.ones(shape) + value - 1
    return  tensor_fill_value
class Model(object):
    
    def __init__(self, dimensional = 2, a = None, b = None, value = 0., learning_rate = 0.1):
        self._x = fill_value_vector(value, tuple((dimensional, )))
        self._dimensional = dimensional
        self._a_const = tf.constant(a, name = 'a', dtype = tf.float32)
        self._b_const = tf.constant(b, name = 'b', dtype = tf.float32)
        self._learning_rate = learning_rate
        self._Energy = tf.reduce_sum(tf.square(self._x))
        
    def __call__(self): # Бесполезная по сути
        return self._x + (self._a_const + self._b_const)
    
    
    def __repr__(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            str_x_ = str(sess.run(self._x))
            str_a = str(sess.run(self._a_const))
            str_b = str(sess.run(self._b_const))
        return str_x_ + " const: (%s, %s)" % ( str_a, str_b )    
    
    def Energy_mutable(self):
        self._Energy = tf.reduce_sum(tf.square(self._x))
    
                
learning_rate = 0.01
model =  Model(dimensional = 3, a = 2., b = 3., value = 1., 
               learning_rate = learning_rate)
show_tf(model._Energy)
def system_functions(model):
    f_1 = tf.add(tf.reduce_sum(tf.square(model._x)), -model._a_const)
    f_2 = tf.add(-tf.reduce_sum(tf.square(model._x)), model._b_const)
    functionals_ = tf.reduce_min([f_1, f_2])
    return functionals_
def train(model):
    with tf.GradientTape() as t:
        t.watch(model._x)
        current_funct_ = system_functions(model)
        grad_x = t.gradient(current_funct_, model._x)
        model._x = model._x + model._learning_rate * grad_x
        model.Energy_mutable()
        
x_list = []
current_energy_tf = []
epochs = range(100)
for epoch in epochs:
    x_list.append((model._x))
    current_funct_ = system_functions(model)
    current_energy_tf.append(model._Energy)
    train(model)
    #print('Epoch %d and X vector: %s' % (epoch, model))
    
current_energy_ = []
current_point = []
def write_list_of_tensors(x_list, current_function):    
    with tf.Session() as sess:
        for item in range(len(x_list)):
            current_energy_.append(sess.run(current_function[item]))
            current_point.append(sess.run(x_list[item]))
            #print(sess.run(x_list[item]))
            #print(sess.run(current_function[item]))
        
write_list_of_tensors(x_list, current_energy_tf)
print(current_energy_)
### built graphs of energy ###
fig, ax = plt.subplots(figsize = (8, 6))
data_x = [a for a in range(len(x_list))]
ax.grid()
ax.plot(data_x, current_energy_, 'b');
ax.plot(data_x, [3. for b in range(len(data_x))], 'r');
ax.plot(data_x, [2. for a in range(len(data_x))], 'r');
ax.legend(['Enegry', 'bound_up', 'bound_down'], loc = 'lower left') # Место расположения легенды
plt.savefig('Energy_1.png', format='png', dpi=100)
plt.clf()

