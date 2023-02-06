import random
import numpy as np
from utils import Chromosome

class Mutation(object):
	def mutate(self):
		return NotImplementedError()


class GausianMutation(Mutation):
	def __init__(self, sigma):
		self.sigma = sigma

	def mutate(self, child, ob_func):
		newGene = child.genes + np.random.randn(len(child.genes))*self.sigma
		fitness = ob_func(*newGene)
		return Chromosome(newGene, fitness)
