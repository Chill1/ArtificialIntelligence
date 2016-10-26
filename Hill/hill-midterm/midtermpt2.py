import numpy as np
import os, subprocess


def question2(N, ep, dvc, delta):
	for x in range(0, 10):
		eq1 = 8 / (ep * ep) 
		eq2 = (4 * (((2 * N) ** dvc) + 1)) / delta
		eq2 = np.log(eq2)
		answer = eq1*eq2
		N = answer
		print answer
	return answer


def main():
	a = question2(N = 1000, dvc = 10, ep = 0.05, delta = 0.05)
	print "Answer after 10 iterations: " + str(a)

main()
