## Genetic algorithm implementation on Python
------------------
This module contains:
* selection.py: all selection methods for choosing parents.
* crossover.py: class contains cross over methods of exchanging segments between parents or choromosomes.
* mutation.py: define mutation methods of a child.
* genetic.py: main implementation of genetic algorithm.

To run example:
```python
from utils import Objective_function
from genetic import GeneticAlgorithm

def func(x,y):
	# McCormick function
	return np.sin(x+y) + (x-y)**2 -1.5*x + 2.5*y + 1
#set contraint for varibles
lo_bounds = [-1.5,-3]
up_bounds = [4,4]
def main():
	# Define objective class
	objective_class = Objective_function(func1, 2,lo_bounds1, up_bounds1)

	#set GA optimizer
	ga = GeneticAlgorithm(nb_generations = 20,
				nb_populations = 10,
				objective_class = objective_class)

	ga.evolve()
```
