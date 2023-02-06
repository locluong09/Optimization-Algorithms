import numpy as np
import math

from utils import Objective_function
from genetic import GeneticAlgorithm

# Define the objective function
def func(x,y):
	# McCormick function
	return np.sin(x+y) + (x-y)**2 -1.5*x + 2.5*y + 1
#set contraint for varibles
lo_bounds = [-1.5,-3]
up_bounds = [4,4]

def func1(x,y):
	# Himmelblau function
	return (x**2 + y -11)**2 + (x+y**2 - 7)**2

lo_bounds1 = [-5,-5]
up_bounds1 = [5,5]

def func2(x,y):
	return (x**2 + y**2)

def main():
	# Define objective class
	objective_class = Objective_function(func1, 2,lo_bounds1, up_bounds1)

	#set PSO optimizer
	ga = GeneticAlgorithm(nb_generations = 20,
				nb_populations = 10,
				objective_class = objective_class)

	ga.evolve()
	for i in ga.populations:
		print(i.fitness)

if __name__ == "__main__":
	main()
