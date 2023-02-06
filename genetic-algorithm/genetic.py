import numpy as np
import random

from mutation import GausianMutation
from selection import TournamentSelection
from crossover import SinglePointCrossover
from utils import Chromosome, Objective_function


class GeneticAlgorithm():
	def __init__(self, nb_generations, nb_populations, objective_class):
		self.nb_populations = nb_populations
		self.nb_generations = nb_generations
		self.objective_class = objective_class
		self.populations = []
		

	def set_bounds(self):
		self.lower = self.objective_class.lo_bounds
		self.upper = self.objective_class.up_bounds

	def set_populations(self, dimensions):
		self.set_bounds()
		for _ in range(self.nb_populations):
			individual = Chromosome()
			individual.set_genes(dimensions, self.lower, self.upper)
			individual.get_fitness(self.objective_class.ob_func)
			self.populations.append(individual)

	def evolve(self):
		self.set_populations(self.objective_class.n_variables)

		ob_func = self.objective_class.ob_func

		selection_method  = TournamentSelection(3)
		crossover_method = SinglePointCrossover()
		mutation_method = GausianMutation(0.5)

		for i in range(self.nb_generations):
			fitness_list = [population.fitness for population in self.populations]
			print(fitness_list)
			selected_parents = [[self.populations[selection_method.select(fitness_list)], self.populations[selection_method.select(fitness_list)]] 
				for _ in range(self.nb_populations)]

			for j in range(self.nb_populations):
				child = crossover_method.cross_over(selected_parents[j][0], selected_parents[j][1], ob_func)
				mutate_rate = np.random.choice([0,1], p = [0.99, 0.01])
				if mutate_rate == 1:
					child = mutation_method.mutate(child, ob_func)

				self.populations[j] = child














