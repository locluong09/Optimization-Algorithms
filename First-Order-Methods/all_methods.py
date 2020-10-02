import numpy as np

class GradientDescent():
	def __init__(self, learning_rate = 0.01):
		self.learning_rate = learning_rate
		self.weight_update = None

	def get_update(self, weight, gradient):
		if self.weight_update is None:
			self.weight_update = np.zeros(np.shape(weight))

		self.weight_update = gradient
		return weight - self.learning_rate*self.weight_update


class StochasticGradientDescent():
	def __init__(self, learning_rate = 0.01, momentum = 0.9, nesterov = False):
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.nesterov = nesterov
		self.weight_update = None


	def get_update(self, weight, gradient):
		if self.weight_update is None:
			self.weight_update = np.zeros(np.shape(weight))

		self.weight_update = self.momentum*self.weight_update + self.learning_rate*gradient
		self.velocity = self.weight_update
		if self.nesterov:
			return weight - (self.momentum*self.velocity + self.learning_rate*self.weight_update)
		else:
			return weight - self.weight_update

class Adagrad():
	def __init__(self, learning_rate = 0.01, epsilon = 1e-6):
		self.learning_rate = learning_rate
		self.epsilon = epsilon
		self.accumulator = None
	def get_update(self, weight, gradient):
		if self.accumulator is None:
			self.accumulator = np.zeros(np.shape(weight))
		self.accumulator += np.square(gradient)
		return weight - self.learning_rate*gradient/(np.sqrt(self.accumulator + self.epsilon))

class Adadelta():
	def __init__(self, rho = 0.9, epsilon = 1e-6):
		self.rho = rho
		self.epsilon = epsilon
		self.accumulator = None
		self.accumulator_delta = None

	def get_update(self, weight, gradient):
		if self.accumulator is None:
			self.accumulator = np.zeros(np.shape(weight))
		if self.accumulator_delta is None:
			self.accumulator_delta = np.zeros(np.shape(weight))

		self.accumulator = self.rho * self.accumulator + (1 - self.rho)*np.power(gradient, 2)
		lr = np.sqrt(self.accumulator_delta + self.epsilon) / np.sqrt(self.accumulator + self.epsilon)
		self.weight_update = lr*gradient
		self.accumulator_delta = self.rho * self.accumulator_delta + (1 - self.rho)*(np.power(self.weight_update, 2))

		return weight - self.weight_update

class RMSProp():
	def __init__(self, learning_rate = 0.01, rho = 0.9, epsilon = 1e-6):
		self.learning_rate = learning_rate
		self.rho = rho
		self.epsilon = epsilon
		self.accumulator = None

	def get_update(self, weight, gradient):
		if self.accumulator is None:
			self.accumulator = np.zeros(np.shape(weight))

		self.accumulator = self.rho * self.accumulator + (1 - self.rho) * np.power(gradient, 2)
		return weight - self.learning_rate * gradient / np.sqrt(self.accumulator + self.epsilon)


class Adam():
	def __init__(self, learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
		self.learning_rate = learning_rate
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.epsilon = epsilon
		self.time_step = 0
		self.ms = None
		self.vs = None
	def get_update(self, weight, gradient):
		if self.ms is None:
			self.ms = np.zeros(np.shape(weight))

		if self.vs is None:
			self.vs = np.zeros(np.shape(weight))

		self.time_step += 1

		self.ms = self.beta_1 * self.ms + (1- self.beta_1) * gradient
		self.vs = self.beta_2 * self.vs + (1- self.beta_2) * np.power(gradient, 2)
		# print(self.time_step)
		# self.learning_rate = self.learning_rate*np.sqrt(1 - self.beta_2** self.time_step) / (1 - self.beta_1**self.time_step)
		# print(self.learning_rate)
		
		return weight - self.learning_rate * self.ms / (np.sqrt(self.vs)+ self.epsilon)

class Adamax():
	def __init__(self, learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
		self.learning_rate = learning_rate
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.epsilon = epsilon
		self.time_step = 0
		self.vt = None
		self.ut = None

	def get_update(self, weight, gradient):
		if self.ms is None:
			self.vt = np.zeros(np.shape(weight))

		if self.vs is None:
			self.ut = np.zeros(np.shape(weight))

		self.time_step += 1

		self.vt = self.beta_1 * self.vt + (1- self.beta_1) * gradient
		self.ut = np.maximum(self.beta_2 * self.ut, np.abs(gradient))

		# self.learning_rate = self.learning_rate*np.sqrt(1 - self.beta_2** self.time_step) / (1 - self.beta_1**self.time_step)
		# print(self.learning_rate)
		
		return weight - self.learning_rate * self.vt/ (self.ut+ self.epsilon)

class NAdam():
	def __init__(self, learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
		self.learning_rate = learning_rate
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.epsilon = epsilon
		self.time_step = 0
		self.mt = None
		self.vt = None

	def get_update(self, weight, gradient):
		if self.ms is None:
			self.mt = np.zeros(np.shape(weight))

		if self.vs is None:
			self.vt = np.zeros(np.shape(weight))

		self.mt = self.beta_1 * self.mt + (1- self.beta_1) * gradient
		self.mt_bar = self.mt/(1 - self.beta_1)

		self.vt = self.beta_2 * self.vt + (1 - self.beta_2) * np.power(gradient, 2)
		self.vt_bar = self.vt/(1 - self.beta_2)

		# self.learning_rate = self.learning_rate*np.sqrt(1 - self.beta_2** self.time_step) / (1 - self.beta_1**self.time_step)
		# print(self.learning_rate)
		
		return weight - self.learning_rate * (self.mt_bar * self.beta_1 + (1 - self.beta_1)/self.beta_1 *gradient)/ (np.sqrt(self.vt_bar)+ self.epsilon)


