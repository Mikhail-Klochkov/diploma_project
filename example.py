
from class_Markovitz import wrapper_fuzzy_constraints_, Markovitz_theory
import tensorflow.compat.v2 as tf
import numpy as np

# Мы должны передать туда в функцию числа по порядку в размере от 2 - num_param
# Оно устанавливает мягкие соотношения упорядоченности между параметрами инвестиционного портфеля
# Количество минимальных аргументов 2 и до num_param.
num_of_parameters = 5
def create_a_list_indeces(max_indeces):
	# последовательно вводим параметры до того, как у нас не прийдёт 0
	flag = True
	indeces_ = []
	while(flag):
		pair_ = [int(a) for a in list(input()) if a != ' ']
		if any([a > max_indeces for a in pair_]):
			assert (False), ('We enter index more then available!')

		print(pair_)
		if(pair_[0] == 0 or pair_[1] == 0):
			flag = False
		else:
			indeces_.append(pair_)

	return indeces_		

#indeces_ = create_a_list_indeces(num_of_parameters)
#print([(iter_1, iter_2) \
#	for iter_1, iter_2 in indeces_])


markov_theory = Markovitz_theory()
markov_theory._initial_model_param_()

funct_obj = wrapper_fuzzy_constraints_(markov_theory)
x = tf.Variable(np.random.random(markov_theory.weights.shape[0]))
print(x.shape)
with tf.GradientTape() as g:
	g.watch((x))
	f = funct_obj(x)
	grad_ = g.gradient(f, x)
	print(grad_.numpy())