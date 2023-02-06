import numpy as np
import math

from utils import Objective_function
from particle_swarm_optimization import PSO

# Define the objective function
def func(x,y):
	# McCormick function
	return np.sin(x+y) + (x-y)**2 -1.5*x + 2.5*y + 1
	
#set contraint for varibles
lo_bounds = [-1.5,-3]
up_bounds = [4,4]
n_variables = 2

def g(x,y):
	# Define this function to find the maximum of f(x,y).
	return -func(x,y)

def main():
	# Define objective class
	objective_class = Objective_function(func, n_variables,lo_bounds, up_bounds)

	#set PSO optimizer
	pso_opt = PSO(nb_generations = 50,
				nb_populations = 100,
				objective_class = objective_class,
				w = 0.72984,
				c1 = 1.496172,
				c2 = 1.496172,
				max_velocity = 20,
				min_velocity = -20)

	pso_opt.evolve()
	pso_opt.plot_gbest()

if __name__ == "__main__":
	main()


