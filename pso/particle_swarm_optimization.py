import numpy as np
import math
import copy
import matplotlib.pyplot as plt

from utils import Particle

class PSO(object):
	'''
	Particle swarm optimization class
	---
	Attribues:
		nb_generations : number of generations of swarm. It means the number of iterations
		nb_populations : the size of swarm (number of particle in swarm)
		objective_class : the objective function defined as class
		w : the inertia weight, range: (0,1)
		c1 : the cognitive weight, range (0,1)
		c2 : the social weight, range (0,1)
		max_velocity : the maximum value of velocity, use to set constraints of veloctity
		min_velocity : the mininum value of velocity, use to set constraints of veloctity
		gbest : the global best swarm so far during training process.
	Methods:
	---
		set_swarm: initialize the swarm with defined populations
		evolve: update global particle gbest of swarm and pbest of each particle after each iteration.
	'''
	def __init__(self, nb_generations, nb_populations, objective_class, w, c1, c2, max_velocity, min_velocity):
		self.nb_generations = nb_generations
		self.nb_populations = nb_populations
		self.objective_class = objective_class
		self.w = w
		self.c1 = c1
		self.c2 = c2
		self.max_velocity = max_velocity
		self.min_velocity = min_velocity
		self.swarm = []
		self.gbest = None

	def set_bounds(self):
		self.lower = self.objective_class.lo_bounds
		self.upper = self.objective_class.up_bounds

	def set_swarm(self, dimensions):
		self.set_bounds()
		for i in range(self.nb_populations):
			particle = Particle(dimensions)

			if hasattr(particle, 'set_position'):
				particle.set_position(self.lower, self.upper)

			if hasattr(particle, 'set_velocity'):
				particle.set_velocity(self.max_velocity, self.min_velocity)

			self.swarm.append(particle)

	def evolve(self):
		#initialize the swarm
		self.set_swarm(self.objective_class.n_variables)

		#set objective function
		ob_class = self.objective_class

		#initialize gbest in swarm
		# PARTICLE = Particle(2)
		# PARTICLE.position = [3.1, 2.1]
		# PARTICLE.best_fitness = 0.01
		self.gbest = copy.copy(self.swarm[0])
		# self.gbest = copy.copy(PARTICLE)
		self.best_list = []

		for i in range(self.nb_generations):
			x_pos, y_pos, best = self.swarm_position()
			self.contour_plot(x_pos, y_pos, best)
			for particle in self.swarm:
				#update particle velocity
				particle.update_velocity(self.w, self.c1, self.c2, self.gbest.position)
				#update particle position
				particle.update_position(self.lower, self.upper)
				#calculate fitness
				particle.calculate_fitness(ob_class)
				#update particle best neighborhoob position and their fitness
				if particle.fitness < particle.best_fitness:
					particle.best_fitness = particle.fitness
					particle.pbest = particle.position
				#update global best particle in the swarm
				if particle.fitness < self.gbest.fitness:
					self.gbest = copy.deepcopy(particle)
				# print(self.gbest.fitness, self.gbest.best_fitness)
				# print(self.gbest.position)
			self.best_list.append(self.gbest.fitness)
			print("At iteration : {}, best position is : {}, with best fitness : {}".format(i+1, self.gbest.position, self.gbest.fitness))

		return self.gbest


	def plot_gbest(self):
		plt.plot(np.arange(self.nb_generations), self.best_list)
		plt.xlabel("Iterations")
		plt.ylabel("Best global paricle fitness")
		plt.title("The optimum point of function versus iterations")
		plt.show()

	def swarm_position(self):
		X = []
		Y = []
		best = self.swarm[0]
		re = []
		for particle in self.swarm:
			X.append(particle.position[0])
			Y.append(particle.position[1])
			if particle.fitness < best.fitness:
				best = particle
		re.append(best.position)
		return X,Y,re

	def contour_plot(self, x_pos, y_pos, best):
		X_coordinate = np.linspace(self.lower[0], self.upper[0])
		Y_coordinate = np.linspace(self.lower[1], self.upper[1])
		X,Y = np.meshgrid(X_coordinate, Y_coordinate)
		Z = self.objective_class.ob_func(X,Y)
		fig = plt.figure(figsize=(6,5))
		left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
		ax = fig.add_axes([left, bottom, width, height]) 

		mycmap = plt.get_cmap('gist_earth')
		cp = plt.contourf(X,Y,Z, cmap = 'gist_earth', extent = (-10,10,-20,20))
		#plt.clabel(cp, inline = True, fontsize = 10)
		#plt.clabel(cp, colors = 'k', fmt = '%2.1f', fontsize=12)
		plt.colorbar(cp)
		plt.scatter(x_pos, y_pos,color = 'k')
		plt.scatter(best[0][0], best[0][1], color = 'r')
		ax.set_title('Contour Plot')
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		plt.ion()
		plt.show()
		plt.pause(0.05)
		# plt.ioff()
		# print("---Plot graph finish---")
		# plt.show()


