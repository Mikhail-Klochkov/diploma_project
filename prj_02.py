import  tensorflow.compat.v2 as tf
import numpy as np
from scipy.optimize import minimize

@tf.function
def print_tensor(tensor):
    tf.print(tensor)


tf_ = tf.constant(3.0, dtype = tf.float32)
print_tensor(tf_)


def function_(x_tensor_):
    print("We generate tensor of functionals:" )
    print("info about tensor " + x_tensor_.dtype.__str__())
    print('shape: ' + x_tensor_.shape.__str__())

    return None

class Functional(tf.Module):

    def tf_to_numpy(tensor): # tensor convert to numpy object
        return tensor.numpy()

    def __init__(self, xtensor_, tuple_of_koeff):
        assert ((xtensor_.shape).__len__() == 1), \
                ('We take a xtensor shape not (ndims, )! \n shape tensor is :{!s}'.format(xtensor_.shape.__len__()))

        self._dimensional = xtensor_.shape[0]
        self._vector = xtensor_
        self._parameter = tf.constant(np.array(list(tuple_of_koeff)), dtype = tf.float32)

        self._vector = tf.reshape(self._vector, [self._dimensional,1])
        self._parameter = tf.reshape(self._parameter, [self._dimensional,1])

       # print(self._vector.__repr__(), self._parameter.__repr__())

        res_tf = tf.map_fn(lambda x: x ** 2, self._vector, dtype = tf.float32)

        self._functional = tf.matmul(res_tf , self._parameter, transpose_a = True)
        self._functional = tf.reshape(self._functional, ())
        del res_tf
        print(self._functional.__repr__())

    def __call__(self):
        return self._functional

    def _change(self, new_vector):
        @tf.function
        def assign_(x_new):
            x_new_tensor = tf.convert_to_tensor(x_new)
            if(x_new_tensor.shape != self._vector.shape):
                assert (1 != 1) ('shape of tensor is not a equal!')
            else:
                self._vector.assign(x_new_tensor)

        assign_(new_vector)



    def compute_gradient(self):
        with tf.GradientTape() as g:
            g.watch((self._vector, self._parameter))
            current_ts = tf.map_fn(lambda x: x ** 2, self._vector, dtype = tf.float32)
            fn_ = tf.matmul(current_ts, self._parameter, transpose_a = True)
            grad_ = g.gradient(fn_, self._vector)

            return grad_

    def compute_hessian(self):
        with tf.GradientTape() as g:
            g.watch((self._vector, self._parameter))
            with  tf.GradientTape() as gg:
                gg.watch((self._vector, self._parameter))
                current_ts = tf.map_fn(lambda x: x ** 2, self._vector, dtype = tf.float32)
                fn_ = tf.matmul(current_ts, self._parameter, transpose_a = True)
                grad_ = gg.gradient(fn_, self._vector)

            grad_grad_ = g.gradient(grad_, self._vector)

        return grad_grad_

    def wrap_functionals(self):
        def inner(vars_x) -> float:
            if(vars_x.shape[0] != self._dimensional):
                assert ( 1 != 1), ('Error size of input vector not equal size of dimensional!')
                return None
            else:
                self._vector = tf.convert_to_tensor(np.array(vars_x, dtype = np.float32).reshape([self._dimensional,1]))
                res_tf = tf.map_fn(lambda x: x ** 2, self._vector, dtype = tf.float32)
                self._functional = tf.matmul(res_tf, self._parameter, transpose_a = True)
                self._functional = tf.reshape(self._functional, ())
                local_value = self._functional.numpy()

            return local_value
        return inner



    def __str__(self):

            return "Dimensional_ is {!s} \n Vector x is : {!s} \n Parameter of Functionals is : {!s} \n Value of Functional: {!s}".format(self._dimensional,
                        Functional.tf_to_numpy(self._vector).__str__(),
                                Functional.tf_to_numpy(self._parameter).__str__(), Functional.tf_to_numpy(self._functional))


class Vector_of_space(object):

    def __init__(self, nd_arr, num_dimension):
        ## we transfer a nd_arr is a numpy array
        self._dim = nd_arr.__len__()
        self._x = tf.Variable(initial_value = nd_arr, dtype  = tf.float32, name = 'vector') # create a vector with initial value which we transfer with ndarray type

    def __str__(self):
        return "vector x is size: {!s}, value is: {!s} ".format(self._dim , Functional.tf_to_numpy(self._x).__str__())

x_numpy = np.array([2. for a in range(1, 4, 1)], dtype = np.float32)
x_tens = Vector_of_space(x_numpy, x_numpy.shape[0])
print(x_tens.__str__())

funct = Functional(x_tens._x,
                   tuple([1,-2, 3]))


print_tensor(funct.compute_gradient())
print(funct)
print_tensor(funct.compute_hessian())
F_ = funct.wrap_functionals()
x_0 = np.array([1, 2, 3])
print(F_(x_0))
res_nelder_mead = minimize(F_, x_0, method = 'nelder-mead',
                           options = {'xtol': 1e-5, 'disp' : True})



print(res_nelder_mead.x)


### another
