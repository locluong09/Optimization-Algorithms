import random
import numpy as np

from utils import Chromosome

class SelectionMethods:
	def select(self):
		return NotImplementedError()


class TruncationSelection(SelectionMethods):
	def __init__(self, k):
		self.k = k

	def select(self, populations):
		return NotImplementedError()

class TournamentSelection(SelectionMethods):
	def __init__(self, k):
		self.k = k

	def select(self, y):
		# fitness_list = [population.fitness for population in populations]
		y  = np.array(y)
		p = np.random.permutation(len(y))
		return p[np.argmin(y[p[0:self.k]])]






