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


####================================================================================================================================================#####
#####this is an integrated code file, you can get either flexible or fixed MEF through changing var "flexible"
##########youcan also change time step through var "step"
####================================================================================================================================================#####



##############################################################========================================================================================####################################
#####################=======================================================define function===========================================================####################################
##############################################################========================================================================================####################################
##########feasibility function, return T/F
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
		
####check feasible and how many violations
def check_feasible(individual):
###Feasibility function for the individual. Returns True if feasible, False otherwise.
###used for checking solution feasibility
	x = individual
	a=0
	b=0
	c=0
	d=0
	for j in range(len(person_id)):
		for i in range(len_time):
			current_charging = CR*sum(x[p]*chargingcons[p]for p in range(j*len_time,j*len_time+i+1))/(60/step)
			current_depleting = sum(erij[p] for p in range(j*len_time, j*len_time+i+1))
			if x[j*len_time+i]*erij[j*len_time+i] > 0: a+=1
			if current_charging + BC[j] < current_depleting: b+=1
#		print(j, [j*len_time, (j+1)*len_time])
#		print(x[j*len_time:(j+1)*len_time])
		tot_charging = sum(x[j*len_time:(j+1)*len_time])*CR*step/60
		tot_depleting = sum(erij[j*len_time:(j+1)*len_time])
		if math.floor(abs(tot_charging - tot_depleting)*10)/10 > CR*0.1*step/60: 
			c+=1
			print('Check feasibility, total charging=', tot_charging, ', total depleting=', tot_depleting, ' at person id', person_id[j])
	for i in range(len_time):
		index_same_time = [p*len_time+i for p in range(0, len(person_id))]
		if CR*sum(x[p]*chargingcons[p]for p in index_same_time)/(60/step) + demand_min[i]/1000 > supply_min[i]/1000: 
			d+=1
			print('time exceeds', i, 'total charging=', CR*sum(x[p]*chargingcons[p]for p in index_same_time)/(60/step), 'total demand, supply=', [demand_min[i]/1000, supply_min[i]/1000])
		
	return a, b, c, d

####define a function that only calculates the actual emissions of one charging plan without penalty
def ghg_cal(individual):
	if flexible == 0:
		print('Fixed hourly MEF required')
	else:
		ghg = 0
		x = individual
		i = 0
		
		index_same_time_cur = [p*len_time+i for p in range(0, len(person_id))]
		index_same_time_pre = [p*len_time+i-1 for p in range(1, len(person_id)+1)]
#		print('current, past = ', index_same_time_cur, index_same_time_pre)
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
#			print('current, past = ', index_same_time_cur, index_same_time_pre)
			tot_charging_cur = sum(x[p]*chargingcons[p]for p in index_same_time_cur)*CR
			tot_charging_pre = sum(x[p]*chargingcons[p]for p in index_same_time_pre)*CR
			G_pre = demand_min[i-1] + tot_charging_pre/1000*(60/step)
			G_cur = demand_min[i] + tot_charging_cur/1000*(60/step)
			if G_pre > G_cur:
				mef = -380.6 + 0.027*G_cur - 0.121*(G_cur - G_pre)
			else:
				mef = -196.3 + 0.019*G_cur + 0.045*(G_cur - G_pre)
			ghg = ghg + CR*sum(x[p]*chargingcons[p]for p in index_same_time_cur)*mef/(60/step)/(0.894*0.91)/1000
	return ghg



