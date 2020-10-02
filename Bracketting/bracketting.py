import numpy as np
import math

class Bracketting(object):
	'''
	Given an unimodality function, the objective is to bracket an interval [a,c] containing the 
	global minimum if we can find the points a < b <c such that f(a) > f(b) < f(c)
	'''
	def __init__(self, func, x0, s =1e-2, k = 2.0):
		self.f = func # objective function
		self.x0 = x0 #initial value
		self.s = s #distance to move
		self.k = k #step size

	def bracketting_minimum(self):
		a, ya = self.x0, self.f(self.x0)
		b, yb = self.x0 + self.s, self.f(self.x0 + self.s)

		#compare ya and ya and check if one of them is bigger than the other
		if yb > ya:
			a, b = b,a
			ya, yb = yb,ya
			self.s = -self.s
		# loop to find the bracket interval
		while True:
			c, yc = b + self.s, self.f(b+ self.s)
			if yc > yb :
				if a < c:
					return (a,c)
				else:
					return (c,a)
			a, ya, b, ya = b, ya, c, ya
			self.s = self.s*self.k


if __name__ == "__main__":
	def func(x):
		return x**3 - 3*x + 1
	x0 = 0
	Bracket = Bracketting(func, x0)
	interval = Bracket.bracketting_minimum()
	print(interval)



