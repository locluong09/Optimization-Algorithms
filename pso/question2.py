import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


from utils import Objective_function
from particle_swarm_optimization import PSO


# Define the objective function
def func(x,y):
	# Himmelblau function
	return (x**2 + y -11)**2 + (x+y**2 - 7)**2

def g(x,y):
	return -func(x,y)


lo_bounds = [-5,-5]
up_bounds = [5,5]

x = np.linspace(-5,5,100)
y = np.linspace(-5,5,100)
X,Y = np.meshgrid(x,y)
Z = func(X,Y)


fig = plt.figure()
ax = plt.axes(projection = '3d')

ax.plot_surface(X,Y,Z, cmap = 'viridis')
ax.set_title("Surface plot of Himmelblau function")
plt.show()



def main():
	# Define objective class
	objective_class = Objective_function(func, 2,lo_bounds, up_bounds)

	#set PSO optimizer
	pso_opt = PSO(nb_generations = 50,
				nb_populations = 20,
				objective_class = objective_class,
				w = 0.72984,
				c1 = 1.496172,
				c2 = 1.496172,
				max_velocity = 20,
				min_velocity = -20)

	pso_opt.evolve()

if __name__ == "__main__":
	main()


