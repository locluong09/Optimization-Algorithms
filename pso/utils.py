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

	def fitness(self,ob_func):
		return self.ob_func(*self.variables)


class Particle():
	'''
	Particle class which describes single particle in swarm.
		Attributes:
		---
		dimensions : dimension of the problem, which is the same with number
			of varibles in objective function
		postion : current position of particle
		velocity : velocity of particle
		pbest : the neighborhoob best of particle
		---
		Methods:
		update_velocity : update new velocity
		update_position : update new position
	References :
		https://nathanrooy.github.io/posts/2016-08-17/simple-particle-swarm-optimization-with-python/
		https://medium.com/analytics-vidhya/implementing-particle-swarm-optimization-pso-algorithm-in-python-9efc2eb179a6

	'''

	def __init__(self, dimensions):
		self.dimensions = dimensions
		self.position = [0 for i in range(self.dimensions)]
		self.velocity = None
		self.fitness = float("inf")
		self.best_fitness = float('inf')

	
	def set_position(self, lower, upper):
		
		for i in range(self.dimensions):
			self.position[i] = np.random.uniform(lower[i], upper[i])
		self.pbest = self.position

	def set_velocity(self, max_velocity, min_velocity):
		self.max_velocity = max_velocity
		self.min_velocity = min_velocity
		if self.velocity is None:
			self.velocity = np.zeros(self.dimensions)

	def calculate_fitness(self, ob_class):
		ob_func = ob_class.ob_func
		self.fitness =  ob_func(*self.position)
		

	def update_velocity(self, w, c1, c2, gbest):
		velocity_new = np.zeros_like(self.position)
		for i in range(self.dimensions):
			r1 = np.random.random()
			r2 = np.random.random()

			inertia_term = w*self.velocity[i]
			cognitive_term = c1*r1*(self.pbest[i] - self.position[i])
			social_term = c2*r2*(gbest[i] - self.position[i])
			velocity_new[i] = inertia_term + cognitive_term + social_term
			velocity_new[i] = np.clip(velocity_new[i], self.min_velocity, self.max_velocity)

		self.velocity = velocity_new
		

	def update_position(self, lower, upper):
		for i in range(self.dimensions):
			self.position[i] += self.velocity[i]
			self.position[i] = np.clip(self.position[i],lower[i], upper[i])
		


