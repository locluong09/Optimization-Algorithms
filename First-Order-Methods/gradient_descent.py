import numpy as np

class GradientDescent():
	def __init__(self, learning_rate = 0.01):
		self.learning_rate = learning_rate

	def get_update(self, x0, gradient):
		return x0 - self.learning_rate*gradient



if __name__ == "__main__":
	f = lambda x : x**3 - 10*x**2 + 4*x + 12
	df = lambda x : 3*x**2 - 20*x + 4
	tol = 1e-6
	GD = GradientDescent(0.00001)
	it = 0
	x = np.array([2])
	while True:
		xold = x
		it += 1
		x = GD.get_update(x, df(x))
		print(x)
		if np.linalg.norm(x-xold, np.inf) < tol:
			break
	print("Minimum value of f(x) is {} at {} after {} iterations".format(f(x), x, it))


	
