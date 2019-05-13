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

####================================================================================================================================================#####
#####this is an integrated code file, you can get either flexible or fixed MEF through changing var "flexible"
##########youcan also change time step through var "step"
####================================================================================================================================================#####
#tts_full_file = open('ttsfull_processed.csv','r')
#colnames, tts_sample = ttsselect_5percent(tts_full_file)
tts_sample = []
tts_sample_file = open('/Users/ran/Documents/Github/charging_data/tts5%_processed_1.csv', 'r')
count = 0
for line in tts_sample_file:
	if count == 0:
		colnames = [x.rstrip() for x in line.split(',')]
#		print (colnames)
		tot_energy = 0
		count += 1
	elif count > 0:
		cols = [x.rstrip() for x in line.split(',')]
		tts_sample.append(cols)
		tot_energy = tot_energy + float(cols[10])

#print('total sample energy consumed:', tot_energy)

tts_sample = np.asmatrix(tts_sample)

col_personid = colnames.index('person_id')
col_energy = colnames.index('energy')
col_time = colnames.index('time')
col_starttime = colnames.index('starttime')
col_endtime = colnames.index('endtime')

#########var settings
flexible = 1 ####setup a turn on/off button to switch between two mef modes
person_id = np.unique(np.array(tts_sample[:,col_personid]))#[0:2] ##travelers id, aka veh id
#print('person_id', person_id)

len_time_1min = 1440  ###total min of a day
step = 60 ###calculate per 1hour
len_time = int(len_time_1min/step)
num_bits = len_time*len(person_id)
CR = 50
BC = np.full((len(person_id), 1), 70, dtype = int) ##battery capacity, here set it as a deterministic value

erij_ini_array = np.zeros((len_time_1min*len(person_id), 1))

for i in range(len(person_id)):
	indices = np.where((tts_sample[:, col_personid])==(person_id[i]))
	persontrips = tts_sample[indices[0],:]
	for t in range(len(persontrips[:, 1])):
		derivator = (persontrips[t, col_endtime].astype(int)) - (persontrips[t, col_starttime].astype(int))
		erij_value = persontrips[t, col_energy].astype(float)/derivator
		erij_ini_array[len_time_1min*i+(persontrips[t, col_starttime].astype(int)):len_time_1min*i+persontrips[t, col_endtime].astype(int)] = erij_value

erij = [] ###erij_ini is er per 1min, erij is er per 5min
chargingcons = [] ######if erij[i] > 0, chargingcons[i] = 0, charging = chargingcons[i]*x[i]
sum_timestep = 0
for i in range(0,(len(erij_ini_array))):
	if (i+1) % step != 0:
		sum_timestep = sum_timestep + erij_ini_array[i]
	elif (i+1) % step == 0:
#		print(i+1)
		sum_timestep = sum_timestep + erij_ini_array[i]
		erij.append(sum_timestep[0])

		if sum_timestep > 0:
			chargingcons.append(0)
		else:
			chargingcons.append(1)
		sum_timestep = 0

#print('sum erij, sum erij_ini_array, num_bits, len of person_id', sum(erij), sum(erij_ini_array), len(erij), len(erij)//24)


#grid_file = open('C:/Ran/Electric_charging/Ontario_Electricity_Supply_Aug2016.csv','r')
grid_file = open('/Users/ran/Documents/Github/charging_data/Ontario_Electricity_Supply_Aug2017.csv','r')
next(grid_file)
demand_min = []
supply_min = []
d = 0
for line in grid_file:
	cols = [x.strip() for x in line.split(',')]
#	for i in range(60): ##range(60) assign demand and supply to each min of one hour
	for i in range(int(60/step)):
		demand_min.append(float(cols[3]))
		supply_min.append(float(cols[2]))
grid_file.close()
charging_tpoint = [i for i in range(len(erij)) if erij[i] > 0] #convert to 1-D
charging_option = [0, 0.1, 0.3, 1]


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
		if CR*sum(x[p]*chargingcons[p] for p in index_same_time)/(60/step) + demand_min[i]*1000 > supply_min[i]*1000: return False
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
			current_charging = CR*sum(x[p]*chargingcons[p] for p in range(j*len_time, j*len_time+i+1))/(60/step)
			current_depleting = sum(erij[p] for p in range(j*len_time, j*len_time+i+1))
#			if x[j*len_time+i]*erij[j*len_time+i] > 0: 
#				p1 = p1 + adding_penalty(x[j*len_time+i]*erij[j*len_time+i])
			if current_charging + BC[j] < current_depleting: 
				p2 = p2 + adding_penalty( - current_charging - BC[j] + current_depleting)
		tot_charging = CR*sum(x[p]*chargingcons[p] for p in range(j*len_time, j*len_time+len_time))/(60/step)
		tot_depleting = sum(erij[p] for p in range(j*len_time, j*len_time+len_time))
		if math.floor(abs(tot_charging - tot_depleting)*10)/10 > CR*0.1*step/60: 
			p3 = p3 + adding_penalty(math.floor(abs(tot_charging - tot_depleting)*10)/10 - CR*0.1*step/60)
	for i in range(len_time):
		index_same_time = [p*len_time+i for p in range(0, len(person_id))]
		if CR*sum(x[p]*chargingcons[p] for p in index_same_time)/(60/step) + demand_min[i]*1000 > supply_min[i]*1000: 
			p4 = p4 + adding_penalty(CR*sum(x[p]*chargingcons[p] for p in index_same_time)/(60/step) + demand_min[i]*1000 - supply_min[i]*1000)
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
		tot_charging_cur = sum(x[p]*chargingcons[p] for p in index_same_time_cur)*CR
		tot_charging_pre = sum(x[p]*chargingcons[p] for p in index_same_time_pre)*CR
		G_pre = demand_min[i-1] + tot_charging_pre/1000*(60/step)
		G_cur = demand_min[i] + tot_charging_cur/1000*(60/step)
		if G_pre > G_cur:
			mef = -380.6 + 0.027*G_cur - 0.121*(G_cur - G_pre)
		else:
			mef = -196.3 + 0.019*G_cur + 0.045*(G_cur - G_pre)
		ghg = ghg + CR*sum(x[p]*chargingcons[p] for p in index_same_time_cur)*mef/(60/step)/(0.894*0.91)/1000
			
		for i in range(1,len_time):
			index_same_time_cur = [p*len_time+i for p in range(0, len(person_id))]
			index_same_time_pre = [p*len_time+i-1 for p in range(0, len(person_id))]
			tot_charging_cur = sum(x[p]*chargingcons[p] for p in index_same_time_cur)*CR
			tot_charging_pre = sum(x[p]*chargingcons[p] for p in index_same_time_pre)*CR
			G_pre = demand_min[i-1] + tot_charging_pre/1000*(60/step)
			G_cur = demand_min[i] + tot_charging_cur/1000*(60/step)
			if G_pre > G_cur:
				mef = -380.6 + 0.027*G_cur - 0.121*(G_cur - G_pre)
			else:
				mef = -196.3 + 0.019*G_cur + 0.045*(G_cur - G_pre)
			ghg = ghg + CR*sum(x[p]*chargingcons[p] for p in index_same_time_cur)*mef/(60/step)/(0.894*0.91)/1000
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
			current_charging = CR*sum(x[p]*chargingcons[p] for p in range(j*len_time,j*len_time+i+1))/(60/step)
			current_depleting = sum(erij[p] for p in range(j*len_time, j*len_time+i+1))
			if x[j*len_time+i]*erij[j*len_time+i] > 0: a+=1
			if current_charging + BC[j] < current_depleting: b+=1
#		print(j, [j*len_time, (j+1)*len_time])
#		print(x[j*len_time:(j+1)*len_time])
		tot_charging = CR*sum(x[p]*chargingcons[p] for p in range(j*len_time, j*len_time+len_time))/(60/step)
		tot_depleting = sum(erij[j*len_time:(j+1)*len_time])
		if math.floor(abs(tot_charging - tot_depleting)*10)/10 > CR*0.1*step/60: 
			c+=1
			print('Check feasibility, total charging=', tot_charging, ', total depleting=', tot_depleting, ' at person id', person_id[j])
	for i in range(len_time):
		index_same_time = [p*len_time+i for p in range(0, len(person_id))]
		if CR*sum(x[p]*chargingcons[p] for p in index_same_time)/(60/step) + demand_min[i]*1000 > supply_min[i]*1000: 
			d+=1
			print('time exceeds', i, 'total charging=', CR*sum(x[p]*chargingcons[p] for p in index_same_time)/(60/step), 'total demand, supply=', [demand_min[i]/1000, supply_min[i]/1000])
		
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
		tot_charging_cur = sum(x[p]*chargingcons[p] for p in index_same_time_cur)*CR
		tot_charging_pre = sum(x[p]*chargingcons[p] for p in index_same_time_pre)*CR
		G_pre = demand_min[i-1] + tot_charging_pre/1000*(60/step)
		G_cur = demand_min[i] + tot_charging_cur/1000*(60/step)
		if G_pre > G_cur:
			mef = -380.6 + 0.027*G_cur - 0.121*(G_cur - G_pre)
		else:
			mef = -196.3 + 0.019*G_cur + 0.045*(G_cur - G_pre)
		ghg = ghg + CR*sum(x[p]*chargingcons[p] for p in index_same_time_cur)*mef/(60/step)/(0.894*0.91)/1000
			
		for i in range(1,len_time):
			index_same_time_cur = [p*len_time+i for p in range(0, len(person_id))]
			index_same_time_pre = [p*len_time+i-1 for p in range(0, len(person_id))]
#			print('current, past = ', index_same_time_cur, index_same_time_pre)
			tot_charging_cur = sum(x[p]*chargingcons[p] for p in index_same_time_cur)*CR
			tot_charging_pre = sum(x[p]*chargingcons[p] for p in index_same_time_pre)*CR
			G_pre = demand_min[i-1] + tot_charging_pre/1000*(60/step)
			G_cur = demand_min[i] + tot_charging_cur/1000*(60/step)
			if G_pre > G_cur:
				mef = -380.6 + 0.027*G_cur - 0.121*(G_cur - G_pre)
			else:
				mef = -196.3 + 0.019*G_cur + 0.045*(G_cur - G_pre)
			ghg = ghg + CR*sum(x[p]*chargingcons[p] for p in index_same_time_cur)*mef/(60/step)/(0.894*0.91)/1000
	return ghg


