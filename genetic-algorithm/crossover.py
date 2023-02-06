import random
import numpy as np
import math

from utils import Chromosome

class Crossover:
	def cross_over(self):
		return NotImplementedError()

	@staticmethod
	def convert_float(num, base = 2, digits=None):
		num = float(num)
		if digits is None: digits = 6

		num = int(round(num * pow(base, digits)))
		num = convert_int(num, base)
		num = num[:-digits] + '.' + num[:digits]
		if num.startswith('.'): num = '0' + num
		return num

	def float_to_bin(num, length):
		integer = math.floor(num)
		dec = num - integer

		bin_int = '{0:bin_int}'.format(integer)
		while len(bin_int) < length:
			bin_int = '0' + bin_int
		


class SinglePointCrossover(Crossover):
	def cross_over(self, parent1, parent2, ob_func):
		crossPoint = random.randrange(0, len(parent1.genes))
		gene1 = parent1.genes[0:crossPoint]
		gene2 = parent2.genes[crossPoint:len(parent2.genes)]
		newGene = np.hstack((gene1, gene2))
		# print(newGene)
		fitness = ob_func(*newGene)
		return Chromosome(newGene, fitness)