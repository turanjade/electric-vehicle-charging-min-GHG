import math
import random

def ap_func(old, new, T):
	if T <= 0: 
		print('Temperature must be positive')
	else: 
		ap = math.exp((- new + old)/T)

	return ap


def select_SA(parent, child, T):
#########produce next gen by SA selection
#####have to define: acceptance function, compare two individuals, one from parent, another from child, at the same position
	
	new_gen = []
	i = 0
	for i in range(len(parent)):
		ind_p = parent[i]
		ind_c = child[i]
		fit_p, = ind_p.fitness.values
		fit_c, = ind_c.fitness.values
		
		if fit_c >= fit_p:
			ap = ap_func(fit_p, fit_c, T)
			if ap >= random.random():
				new_gen.append(ind_c)
			else:
				new_gen.append(ind_p)
		elif fit_c < fit_p:
			new_gen.append(ind_c)
	
	return new_gen
	
