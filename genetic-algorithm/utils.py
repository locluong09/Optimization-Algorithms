import numpy as np
import math

class Objective_function():
	'''
	Define an objective function as class
	---
	Atributes:
		ob_func : objective function
		n_varibales : number of variables of objective function
		up_bounds : list containing the uppper bound of  variables
		lo_bounds : list containing the lower bound of  variables
	Methods:
		set_bounds : set upper and lower bounds
		fitness : return output of objective function
	'''
	def __init__(self, ob_func, n_variables, lo_bounds, up_bounds):
		self.ob_func = ob_func
		self.n_variables = n_variables
		
		self.lo_bounds = lo_bounds
		self.up_bounds = up_bounds

	def set_variables(self, n_variables):
		self.variables = []
		for i in n_variables:
			self.variables.append(np.random.random())

	def set_bounds(self):
		if self.up_bounds is None:
			for i in range(len(self.n_variables)):
				self.up_bounds[i] = np.Inf

		if self.lo_bounds is None:
			for i in range(len(self.n_variables)):
				self.lo_bounds[i] = -np.NINF

	def get_fitness(self,ob_func):
		return self.ob_func(*self.variables)


class Chromosome:
	def __init__(self, genes = 0, fitness = 0):
		self.genes = genes
		self.fitness = fitness

	# @property
	def get_fitness(self, ob_func):
		fitness = ob_func(*self.genes)
		self.fitness = fitness

	def set_genes(self, dimensions, lower, upper):
		self.genes = np.random.uniform(lower, upper, dimensions)






