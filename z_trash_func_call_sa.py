import random
import numpy as np
import math
import random
from deap import base, creator, tools, algorithms
from operator import attrgetter
from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import cpu_count as cpucount
import time
import multiprocessing
import functools
from func_varOr import varOr ##inputs include: population, toolbox, lambda_, cxpb, mutpb
from func_nextGen_SA import SimulatedAnnealing
from func_tripselect import ttsselect_5percent 




def sa_evolution_deap(writefiledir, person_id, flexible, step, len_time, num_bits, CR, BC, erij, chargingcons, demand_min, supply_min):
	
	def feasible(individual):
	###Feasibility function for the individual. Returns True if feasible, False otherwise.
		x = individual
		
	#	print(x[288])
		for j in range(len(person_id)):
			for i in range(len_time):
	#			print (j*len_time,j*len_time+i+1)
				current_charging = CR*sum(x[p]*chargingcons[p] for p in range(j*len_time,j*len_time+i+1))/(60/step)
				current_depleting = sum(erij[p] for p in range(j*len_time, j*len_time+i+1))
	#			if x[j*len_time+i]*erij[j*len_time+i] > 0: return False
				if current_charging + BC[j] < current_depleting: return False
			tot_charging = CR*sum(x[p]*chargingcons[p] for p in range(j*len_time, j*len_time+len_time))/(60/step)
			tot_depleting = sum(erij[p] for p in range(j*len_time, j*len_time+len_time))
			if math.floor(abs(tot_charging - tot_depleting)*10)/10 > CR*0.1*step/60: return False
		for i in range(len_time):
			index_same_time = [p*len_time+i for p in range(0, len(person_id))]
			if CR*sum(x[p]*chargingcons[p]for p in index_same_time)/(60/step) + demand_min[i]*1000 > supply_min[i]*1000: return False
		return True

	def adding_penalty(a):
		if abs(a)>=1: p = a**2
		elif abs(a)<1: p = math.log(abs(a)+1)
		return p
	def penalty_direct(individual):
		x = individual
		p1 = 0 ####restrict driving vs charging conflict
		p2 = 0 ####restrict battery capacity conflict
		p3 = 0 ####restrict equilibrium
		p4 = 0 ####restrict grid capacity
		for j in range(len(person_id)):
			for i in range(len_time):
				current_charging = CR*sum(x[p]*chargingcons[p]for p in range(j*len_time, j*len_time+i+1))/(60/step)
				current_depleting = sum(erij[p] for p in range(j*len_time, j*len_time+i+1))
	#			if x[j*len_time+i]*erij[j*len_time+i] > 0: 
	#				p1 = p1 + adding_penalty(x[j*len_time+i]*erij[j*len_time+i])
				if current_charging + BC[j] < current_depleting: 
					p2 = p2 + adding_penalty( - current_charging - BC[j] + current_depleting)
			tot_charging = CR*sum(x[p]*chargingcons[p]for p in range(j*len_time, j*len_time+len_time))/(60/step)
			tot_depleting = sum(erij[p] for p in range(j*len_time, j*len_time+len_time))
			if math.floor(abs(tot_charging - tot_depleting)*10)/10 > CR*0.1*step/60: 
				p3 = p3 + adding_penalty(math.floor(abs(tot_charging - tot_depleting)*10)/10 - CR*0.1*step/60)
		for i in range(len_time):
			index_same_time = [p*len_time+i for p in range(0, len(person_id))]
			if CR*sum(x[p]*chargingcons[p]for p in index_same_time)/(60/step) + demand_min[i]*1000 > supply_min[i]*1000: 
				p4 = p4 + adding_penalty(CR*sum(x[p]*chargingcons[p]for p in index_same_time)/(60/step) + demand_min[i]*1000 - supply_min[i]*1000)
		return [p1, p2, p3, p4]


	##decorate function, used only in toolbox.evaluate
	def min_ghg_MEF(individual):
		penalty = 0
		if feasible(individual) is False:
			penalty = sum(penalty_direct(individual)) + 20000 ###have to add a big number so that violating solutions always have worse fitness
		if flexible == 0:
			print('Fixed hourly MEF required')
		else:
			ghg = 0
			x = individual
			i = 0	
			index_same_time_cur = [p*len_time+i for p in range(0, len(person_id))]
			index_same_time_pre = [p*len_time+i-1 for p in range(1, len(person_id)+1)]
			tot_charging_cur = sum(x[p]*chargingcons[p]for p in index_same_time_cur)*CR
			tot_charging_pre = sum(x[p]*chargingcons[p]for p in index_same_time_pre)*CR
			G_pre = demand_min[i-1] + tot_charging_pre/1000*(60/step)
			G_cur = demand_min[i] + tot_charging_cur/1000*(60/step)
			if G_pre > G_cur:
				mef = -380.6 + 0.027*G_cur - 0.121*(G_cur - G_pre)
			else:
				mef = -196.3 + 0.019*G_cur + 0.045*(G_cur - G_pre)
			ghg = ghg + CR*sum(x[p]*chargingcons[p]for p in index_same_time_cur)*mef/(60/step)/(0.894*0.91)/1000
				
			for i in range(1,len_time):
				index_same_time_cur = [p*len_time+i for p in range(0, len(person_id))]
				index_same_time_pre = [p*len_time+i-1 for p in range(0, len(person_id))]
				tot_charging_cur = sum(x[p]*chargingcons[p]for p in index_same_time_cur)*CR
				tot_charging_pre = sum(x[p]*chargingcons[p]for p in index_same_time_pre)*CR
				G_pre = demand_min[i-1] + tot_charging_pre/1000*(60/step)
				G_cur = demand_min[i] + tot_charging_cur/1000*(60/step)
				if G_pre > G_cur:
					mef = -380.6 + 0.027*G_cur - 0.121*(G_cur - G_pre)
				else:
					mef = -196.3 + 0.019*G_cur + 0.045*(G_cur - G_pre)
				ghg = ghg + CR*sum(x[p]*chargingcons[p]for p in index_same_time_cur)*mef/(60/step)/(0.894*0.91)/1000
		return ghg+penalty,

		
	###decorate individual, mate and mutation process: driving cannot be the same time point as charging
	charging_tpoint = [i for i in range(len(erij)) if erij[i] > 0] #convert to 1-D
	charging_option = [0, 0.1, 0.3, 1]
	creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
	creator.create('Individual', list, fitness=creator.FitnessMin)
	###initiate toolbox
	toolbox = base.Toolbox()
	toolbox.register('attr_rate', random.choice, [0, 0.1, 0.3, 1]) ###function to generate individual
	toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_rate, num_bits)
	toolbox.register('population', tools.initRepeat, list, toolbox.individual) ###tools to generate population
	toolbox.register('evaluate', min_ghg_MEF)

	toolbox.register('mate', tools.cxUniform, indpb=0.3)
	toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.5)
	toolbox.register('select', tools.selTournament, tournsize=int(num_bits))
		
	
	nind = [200]#, 150, 200, 250, 300, 350, 400]
#	nind = [int(num_bits/2)]
	T = 1
	T_min = 0.1
	alpha = 0.9
	
	best_each_rand = []
	
	for i in range(2):
		for n in nind:
			
			randomseeds = random.randint(100,10000)
			random.seed(randomseeds) ###have to change it then see which num_generation and num_ind generates the least std.dev

			pool = multiprocessing.Pool()
			toolbox.register('map', pool.map)			
						
			num_individual = n * 5
			num_gen_eachT = 100
			lambda_ = n * 5 ###in the modified SA, since parent and child will compare their individuals one-by-one, so the total number of ind in both sets shoud be equal
			probab_crossing, probab_mutating = 0.7, 0.05
			
			population = toolbox.population(n=num_individual) ###argument n, how many individuals in the population
			stats = tools.Statistics(key=lambda ind: ind.fitness.values)
			stats.register("avg", np.mean)
			stats.register("std", np.std)
			stats.register("min", np.min)
			stats.register("max", np.max)
			
			print('\nEvaluation process starts')
			pop, log = SimulatedAnnealing(population, toolbox, lambda_, probab_crossing, probab_mutating, \
				num_gen_eachT, T, T_min, alpha, stats, verbose=True)		
						
			pool.close()
			pool.join()	
			
			best_ind_pop = tools.selBest(pop, 1)[0]
			
			optimal_file = open(writefiledir+'/TTSSample_flexible_cx0.7mut0.05_ngen'+str(num_gen_eachT)+'lambda'+str(lambda_)+'_randomseeds'+str(randomseeds[i])+'.txt','w')
			for x in best_ind_pop:
				optimal_file.write(str(x)+'\n')
			optimal_file.close()
			
			best_each_rand.append(best_ind_pop)
	
	return best_each_rand