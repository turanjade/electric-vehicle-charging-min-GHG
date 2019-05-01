from func_result_evaluate_0 import min_ghg_MEF, feasible, ghg_cal
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
from func_singlearray_vehtrip_pars import convert_array

tts_sample = []
#tts_sample_file = open('tts5%_processed_0.csv', 'r')
tts_sample_file = open('/Users/ran/Documents/Github/charging_data/tts5%_processed_0.csv', 'r')
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
print('trip data logged in')

tts_sample = np.asmatrix(tts_sample)

col_personid = colnames.index('person_id')
col_energy = colnames.index('energy')
col_time = colnames.index('time')
col_starttime = colnames.index('starttime')
col_endtime = colnames.index('endtime')
col_origzone = colnames.index('purp_orig')
col_destzone = colnames.index('purp_dest')

#########var settings
flexible = 1 ####setup a turn on/off button to switch between two mef modes
person_id = np.unique(np.array(tts_sample[:,col_personid]))#[0:4] ##travelers id, aka veh id
#print('person_id', person_id)

len_time_1min = 1440  ###total min of a day
step = 60 ###calculate per 1hour
len_time = int(len_time_1min/step)
num_bits = len_time*len(person_id)
grouped = 1 ######define how many persons are grouped as a whole to enter the system each time
num_bits = 1440/step * grouped

CR = 50
BC = np.full((grouped, 1), 70, dtype = int) ##battery capacity, here set it as a deterministic value

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
print('energy data prepared')

###def convert_array(person_id, tts_sample, col_to_convert_orig, col_to_convert_dest, step, col_personid, col_starttime, col_endtime) self-defined function, generate zone list
zonetype = convert_array(person_id, tts_sample, col_origzone, col_destzone, step, col_personid, col_starttime, col_endtime)
print(len(erij), len(zonetype))

exit()

#############base case scenario
#####all level1
##for each trip, charge when it arrives home, if energy has consumed
base_lv1=[]
for i in range(len(person_id)):
	er_person = erij[i*len_time:(i+1)*len_time] ##energy consumption of one tour
	print('length of er_person:',len(er_person))
	ch_person = [0 for p in range(len(er_person))]
	#np.zeros((len(er_person),1),dtype=float).flatten().tolist() ###charging profile
	zonetype_person = zonetype[i*len_time:(i+1)*len_time]
	print('length of zonetype person:', len(zonetype_person))
	
	for t in range(len(er_person)):
		if er_person[t] != 0:
			continue
		if sum(er_person[0:t])-sum(ch_person[0:t])*CR*step/60 <= CR*0.1:#*step/60:
			continue
		elif sum(er_person[0:t])-sum(ch_person[0:t])*CR*step/60 > CR*0.1 and zonetype_person[t] == 'H':#*step/60: #and sum(ch_person[0:t])*CR*step/60 <= sum(er_person[0:t])
			ch_person[t] = 0.1
	base_lv1.extend(ch_person)
print('base case: only level 1 charging, w/w.o. penalty: ', round(min_ghg_MEF(base_lv1),2), round(ghg_cal(base_lv1),2))

###
exit()
###

base_lv2=[]
for i in range(len(person_id)):
	er_person = erij[i*len_time:(i+1)*len_time]
	ch_person = np.zeros((len(er_person),1),dtype=float).flatten().tolist()
	zonetype_person = zonetype[i*len_time:(i+1)*len_time]
	for t in range(len(er_person)):
		if er_person[t] != 0:
			continue
		if sum(er_person[0:t])-sum(ch_person[0:t])*CR*step/60 <= CR*0.1:#*step/60:
			continue
		elif sum(er_person[0:t])-sum(ch_person[0:t])*CR*step/60 > CR*0.1 and sum(er_person[0:t])-sum(ch_person[0:t])*CR*step/60 <= CR*0.3 and sum(ch_person[0:t])*CR*step/60 <= sum(er_person[0:t]) and zonetype_person[t] == 'H':#*step/60:
			ch_person[t] = 0.1
		elif sum(er_person[0:t])-sum(ch_person[0:t])*CR*step/60 > CR*0.3 and sum(ch_person[0:t])*CR*step/60 <= sum(er_person[0:t]) and zonetype_person[t] == 'H':
			ch_person[t] = 0.3
	base_lv2.extend(ch_person)
print('base case: lv1 and lv2 charging: ', round(min_ghg_MEF(base_lv2),2), round(ghg_cal(base_lv2),2))

base_lv3=[]
for i in range(len(person_id)):
	er_person = erij[i*len_time:(i+1)*len_time]
	ch_person = np.zeros((len(er_person),1),dtype=float).flatten().tolist()
	zonetype_person = zonetype[i*len_time:(i+1)*len_time]
	for t in range(len(er_person)):
		if er_person[t] != 0:
			continue
		if sum(er_person[0:t])-sum(ch_person[0:t])*CR*step/60 <= CR*0.1:#*step/60:
			continue
		elif sum(er_person[0:t])-sum(ch_person[0:t])*CR*step/60 > CR*0.1 and sum(er_person[0:t])-sum(ch_person[0:t])*CR*step/60 <= CR*0.3 and sum(ch_person[0:t])*CR*step/60 <= sum(er_person[0:t]) and zonetype_person[t] == 'H':#*step/60:
			ch_person[t] = 0.1
		elif sum(er_person[0:t])-sum(ch_person[0:t])*CR*step/60 > CR*0.3 and sum(er_person[0:t])-sum(ch_person[0:t])*CR*step/60 <= CR and sum(ch_person[0:t])*CR*step/60 <= sum(er_person[0:t]) and zonetype_person[t] == 'H':
			ch_person[t] = 0.3
		elif sum(er_person[0:t])-sum(ch_person[0:t])*CR*step/60 > CR and sum(ch_person[0:t])*CR*step/60 <= sum(er_person[0:t]) and zonetype_person[t] == 'H':
			ch_person[t] = 1
	base_lv3.extend(ch_person)
print('base case: lv1, lv2, and lv3 charging: ', round(min_ghg_MEF(base_lv3),2), round(ghg_cal(base_lv3),2))
