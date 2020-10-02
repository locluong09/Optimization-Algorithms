from __future__ import print_function, division
import numpy as np
import math

class Fibonacci_Search():
	'''
	Given a unimodal function f that is bracketted by the interval [a,b]
	Fibonaccy search will shink the current interval [a,b] to a new interval [a1,b1] which is significanly
	shrunk
	Fn    phi*(1-S^(n+1))
	----=-----------------
	Fn-1	1-s^(n)
	'''
	def __init__(self, n, epsilon):
		self.s = (1-math.sqrt(5))/(1+math.sqrt(5))
		self.golden_ratio = (1+math.sqrt(5))/2
		self.n = n #number of sequences to shink the interval
		self.epsilon = epsilon

	def search(self, f, a, b):
		rho = 1 / (self.golden_ratio*(1-self.s**(self.n+1))/(1-self.s**self.n))
		d = rho*b + (1-rho)*a
		yd = f(d)
		for i in range(self.n):
			if i == self.n:
				c = self.epsilon*a + (1-self.epsilon)*d
			else:
				c = rho*a + (1-rho)*b

			yc = f(c)

			if yc < yd:
				b, d, yd = d, c, yc
			else:
				a, b = b, c
			rho = 1 / (self.golden_ratio*(1-self.s**(self.n-i+1))/(1-self.s**(self.n-i)))

		if a < b:
			return (a,b)
		else:
			return (b,a)


if __name__ == "__main__":
	def func(x):
		return x**3 - 3*x + 1
	x0 = 0
	#func is not unimodal
	Fibonacci = Fibonacci_Search(100, 1e-2)
	f = lambda x : x**3 - 10*x**2 + 4*x + 12
	interval = Fibonacci.search(func,-10,10)
	interval1 = Fibonacci.search(f,4,9)
	print(interval)
	print(interval1)
	print(f(interval1[0]))



