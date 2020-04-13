

import quandl
import pandas as pd

import tensorflow.compat.v2 as tf
from scipy.optimize import minimize
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint

#############
key_const = 'f8nzs159xkB-_NJg4LCB'
dict_of_securities = ['WMT', 'GE', 'TSLA'] 
#dict_of_securities = ['CNP', 'F', 'WMT', 'GE', 'TSLA'] 
quandl.ApiConfig.api_key = key_const
trading_days = 252
#############


## Надо его адаптировать под класс функций, чтобы формат записи был один и тот же ## 
## У нас для создания объекта класса function необходим массив строчный типа tf.Variable
## Ф также функция с сигнатурой foo(tensor_type_obj) : -> tensor_type_obj для нашего функционала
## Пока без ограничений при этом мы передадим в наш класс только initial_value для дальнейшего изменения

class Markovitz_theory(object):
    global key_const # my key for access to data
    global dict_of_securities
    
    def __init__(self):
        
        self.flag = False 
        self.level_risks = None
        self.weights = None
        self.cov_daily_tf = None
        self.returns_annual_tf = None
        
        self.data = quandl.get_table('WIKI/PRICES', ticker = dict_of_securities,
                        qopts = { 'columns': ['date', 'ticker', 'adj_close'] },
                        date = { 'gte': '2014-1-1', 'lte': '2016-12-31' }, paginate = True)
        self.data = self.data.set_index('date')
        self.data = self.data.pivot(columns = 'ticker')
    
    def _initial_model_param_(self):
        if self.flag == False:
            
            num_of_securities = dict_of_securities.__len__()
            daily_returns_data = self.data.pct_change()
            
            self.returns_annual_tf = tf.constant((daily_returns_data.mean() * trading_days)[:, np.newaxis], dtype = tf.float64) # maybe should be a dtype = tf.float32/tf.float64
            self.cov_daily_tf = tf.constant((daily_returns_data.cov() * trading_days).values, dtype = tf.float64) # matrix of covariation between different columns
            
            #cov_annual = cov_daily.mean() * trading_days # Это пока не используется
            ## initial weigthts ## По простому это наши неизвестные, по которым у нас распределены доходы
            ##  он даёт только положительные веса, поэтму  в принципе стоит добавить ограничения
            weights = np.random.random(num_of_securities) # Мы тут выбираем равномерное распределение [0,1]
            
            self.weights = tf.Variable((weights / np.sum(weights))[:, np.newaxis], dtype = tf.float64, name = 'weights')      
            self.flag = True
            print("initial_weights w: {!s}".format(self.weights.numpy())) # call only one time
    
    def _Markovitz_functionals_(self): # weights should be a tensor для реализации functionS 
        ### Здесь мы проинтициализируем параметры модели Марковица для определения функционала(!уровня риска!) - мы его минимизируем
        if(self.flag == False): # Я не задад параметры и начальные веса модели, то мы их инициализируем 
            self._initial_model_param_()
        ### тут мы обменяем два тензорных объекта self.weights <-> weights_tf ###
        ### weights_tf - should be a shape (num, ) -> ###
        return tf.sqrt(tf.matmul(tf.transpose(self.weights), tf.matmul(self.cov_daily_tf, self.weights)))[0][0]
    
    
    def __str__(self):
        return "r: \n {!s} \n W:\n {!s} \n COV(r_i, r_j):\n {!s} \n".format(self.returns_annual_tf.numpy(), self.weights.numpy(),
                                                                 self.cov_daily_tf.numpy())
    # наша модель включает ограничения только типа равенств и неравенств линейного типа. Данная функция будет принимать нижнею
    # Границу наших рисков и минимизировать deviation
    def _linear_constraints_(self, Q):
        linear_constraints = LinearConstraint([list(self.returns_annual_tf.numpy()[:, 0]), # вроде в виде list мы должны засунуть
                                               list(np.ones(self.returns_annual_tf.shape[0]))],
                                               [Q, 1.], [np.inf, 1.])
        return linear_constraints
    def plot_a_convex_field(self, stack_data = None):
        if(self.flag == False):
            self._initial_model_param_()
        
        port_returns = []
        port_volatility = []
        stock_weights = []
        
        num_asset = dict_of_securities.__len__()# for five a securities
        num_portfolios = 50000 # number of generations of portfolios
        
        for single_portfolio in range(num_portfolios):
            weights = np.random.random(num_asset)
            weights = weights / np.sum(weights)
            weights = weights[:, np.newaxis]
            returns = np.dot(weights.T, self.returns_annual_tf.numpy())[0][0]
            volatility  = np.sqrt(np.dot(weights.T, np.dot(self.cov_daily_tf.numpy(), weights))) # В статьке было cov_annual
            port_returns.append(returns)
            port_volatility.append(volatility[0][0])
            stock_weights.append(weights)
            
        portfolio = {'Returns': port_returns,
             'Volatility': port_volatility}    

        for counter, name_access in enumerate(dict_of_securities):
            portfolio[name_access + str('weights')] = [Weights[counter] for Weights in stock_weights]
        ## Create a our data set for visualization ##
        df = pd.DataFrame(portfolio)
        column_name = ['Returns', 'Volatility'] + [stock + 'weights' for stock in dict_of_securities]
        df = df[column_name]
        print(df.head())
        plt.style.use('seaborn')
        df.plot.scatter(x = 'Volatility', y = 'Returns', figsize = (10, 8), grid = True)
        # после опитимизации мы можем для текущего уровня построить точку нашего удовлетворяющего портфеля
        if(stack_data is not None):
            for stack_item in stack_data:
                plt.plot(np.linspace(0.13, 0.37, 50), [stack_item[0] for a in range(50)],'r--', linewidth = 0.5)
                plt.scatter(stack_item[1], stack_item[2], c = 'y', marker = '*', linewidths = 3.)
        else:
            plt.plot(np.linspace(0.13, 0.37, 50), [self.level_risks for a in range(50)],'r--', linewidth = 0.5)
            plt.scatter(self._volatility_().numpy(), self.__return_total__().numpy(), c = 'y', marker = '*', linewidths = 3.)

        plt.xlabel('Volatility (Std. Deviation)')
        plt.ylabel('Expected Returns')
        plt.title('Efficient Frontier')
        plt.savefig('markovitz.png')
        
    def _bounds_(self):
        bounds = Bounds([0.0] * self.weights.shape[0], [np.inf] * self.weights.shape[0])
        return bounds
    
    def __return_total__(self):
        return tf.matmul(tf.transpose(self.weights), self.returns_annual_tf)[0][0]
    
    def _volatility_(self):
        return tf.sqrt(tf.matmul(tf.transpose(self.weights), tf.matmul(self.cov_daily_tf, self.weights)))[0][0]
   
def wrapper_fuzzy_constraints_(model_mark):
	cov_matrix = model_mark.cov_daily_tf
	max_indeces = model_mark.weights.shape[0]
	def _stack_indeces_(max_value_index):
			print("please enter a fuzzy indeces: \n")
			flag = True
			indeces = []
			while(flag):
				pair = [int(a) for a in list(input()) if a != ' ']
				if any([a > max_value_index for a in pair]):
					assert (False), ('We enter index more then available!')
				print(pair)
				if(pair[0] == 0 or pair[1] == 0):
					print("Boom!")
					flag = False
				else:
					indeces.append(pair)
			return indeces
	stack_indeces = _stack_indeces_(max_indeces)

	def _fuzzy_constraints_(weights_tf):
		# Так выглядит stack_indeces [[a1, b1], [a2, b2] ... ] end if an == 0  or bn == 0
		if(weights_tf.shape.__len__() != 2): # Это веса относящиеся только по отношению к модели макровица и вкладов к каждой инвестиции портфеля
			weights_tf = weights_tf[:, np.newaxis]

		# тут нужно учесть мягкие неравенства в виде новых ограничений
		# stack индексов должен быть на единицу больше по модулю
		fuzzy_functional = -tf.reduce_min([weights_tf[iter_1 - 1] - weights_tf[iter_2 - 1] \
											for iter_1 , iter_2 in stack_indeces])
		markovitz_functionals = tf.matmul(tf.transpose(weights_tf), tf.matmul(cov_matrix, weights_tf))[0][0] # [0, 0] чтобы после произведения взять единственный элемент от матрицы [1, 1]
		####### Можно сделать два вида функционалов в теории множественной оптимизации по нескольким целевым функциям
		lambda_weight = tf.random.uniform(shape = (2,), minval = 1., maxval  = 2.)
		linear_functional = markovitz_functionals * lambda_weight[0].numpy() + fuzzy_functional * lambda_weight[1].numpy()
		##### Chebyshev functionals ####
		chebyshev_flag = True
		if(chebyshev_flag):
			epsilon = 1. # Непонятно как его подбирать из каких соображений
			lambda_weight = tf.random.uniform(shape = (2,), minval = 1., maxval  = 2.)
			chebyshev_f = tf.reduce_min([lambda_weight[0].numpy() * fuzzy_functional, lambda_weight[1].numpy() * markovitz_functionals]) + \
													tf.constant(epsilon).numpy() * tf.reduce_sum([fuzzy_functional, markovitz_functionals])
		return chebyshev_f
	return _fuzzy_constraints_

def _wrapper_marcovitz(model_mark):
    cov_matrix = model_mark.cov_daily_tf
    def _Marcovitz_risks(weights_tf):
        if(weights_tf.shape.__len__() != 2):
            weights_tf = weights_tf[:, np.newaxis]

        return tf.matmul(tf.transpose(weights_tf), tf.matmul(cov_matrix, weights_tf))[0][0]
    return _Marcovitz_risks