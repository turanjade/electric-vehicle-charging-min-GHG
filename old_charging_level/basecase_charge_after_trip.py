import random
import numpy as np
import math
import random
from operator import attrgetter
import time
import functools
from func_timebased_overall_hourly import time_base

tts_sample = []
#tts_sample_file = open('tts5%_processed_0.csv', 'r')
num_sample = str(2)
tts_sample_file = open('/Users/ran/Documents/Github/charging_data/tts5%_processed_triple'+num_sample+'.csv', 'r')
#########var settings
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
person_id = np.unique(np.array(tts_sample[:,col_personid]))#[0:4] ##travelers id, aka veh id
#print('person_id', person_id)

len_time_1min = 1440  ###total min of a day
step = 60 ###calculate per 1hour
len_time = int(len_time_1min/step)
num_bits = len_time*len(person_id)
grouped = 1 ######define how many persons are grouped as a whole to enter the system each time
num_bits = 1440/step * grouped

CR = 50
BC = np.full((len(person_id)/grouped, 1), 70, dtype = int) ##battery capacity, here set it as a deterministic value

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
print('energy data prepared, total energy is ', sum(erij))

################define a function to test how many travelers that do not have enough charging
def feasible(individual):
	###Feasibility function for the individual. Returns True if feasible, False otherwise.
	x = individual
	
	#	print(x[288])
	outofenergy = []
	shortofcharge = []
	toomuchcharge = []
	for j in range(len(person_id)):
		for i in range(len_time):
			#			print (j*len_time,j*len_time+i+1)
			current_charging = CR*sum(x[p]*chargingcons[p] for p in range(j*len_time,j*len_time+i+1))/(60/step)
			current_depleting = sum(erij[p] for p in range(j*len_time, j*len_time+i+1))
			#			if x[j*len_time+i]*erij[j*len_time+i] > 0: return False
			if current_charging + BC[j] < current_depleting:
				if person_id[j] not in outofenergy:
					outofenergy.append(person_id[j])
		tot_charging = CR*sum(x[p]*chargingcons[p] for p in range(j*len_time, j*len_time+len_time))/(60/step)
		tot_depleting = sum(erij[p] for p in range(j*len_time, j*len_time+len_time))
		if math.floor((tot_charging - tot_depleting)*10)/10 > CR*0.1*step/60:
			if person_id[j] not in toomuchcharge:
				toomuchcharge.append(person_id[j])
		elif math.floor((tot_charging - tot_depleting)*10)/10 < CR*0.1*step/60:
			if person_id[j] not in shortofcharge:
				shortofcharge.append(person_id[j])
	print('number of travelers who (1).run out of energy during the day; (2).charging-depleting>charging unit; (3).depleting-charging>charging unit')
	return [len(outofenergy), len(toomuchcharge), len(shortofcharge)]

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
charging_option = [0,0.1,0.3,1]

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
		if mef <= 0: mef = 0
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
			if mef <= 0: mef = 0
			ghg = ghg + CR*sum(x[p]*chargingcons[p] for p in index_same_time_cur)*mef/(60/step)/(0.894*0.91)/1000
	return ghg

#############base case scenario
#####all level1
##for each trip, determine the time to charge
base_lv1=[]
#print(np.argwhere(person_id==str(10004502)).flatten()[0])
#print(person_id)
for i in range(len(person_id)):#[(np.argwhere(person_id==str(10004502)).flatten()[0])]:
	#	print(i)
	er_person = erij[i*len_time:(i+1)*len_time]
	ch_person = np.zeros((len(er_person),1),dtype=float).flatten().tolist()
	#	print(len(er_person))
	for t in range(len(er_person)):
		if er_person[t] != 0:
			continue
		if sum(er_person[0:t])-sum(ch_person[0:t])*CR*step/60 < CR*0.1:
		  #			print(sum(ch_person[0:t]), sum(er_person[0:t]))
			continue
		if sum(er_person[0:t])-sum(ch_person[0:t])*CR*step/60 >= CR*0.1 and sum(ch_person[0:t])*CR*step/60 <= sum(er_person[0:t]):
			ch_person[t] =0.1
	base_lv1.extend(ch_person)
print('#################################################################')
print('base case: only level 1 charging (w.o penalty): ', round(ghg_cal(base_lv1),2), 'total charged energy', sum(base_lv1)*CR)
#print(feasible(base_lv1))
print(time_base(base_lv1))

base_lv2=[]
for i in range(len(person_id)):
	er_person = erij[i*len_time:(i+1)*len_time]
	ch_person = np.zeros((len(er_person),1),dtype=float).flatten().tolist()
	#	print(ch_person)
	for t in range(len(er_person)):
		if er_person[t] != 0:
			continue
		elif sum(er_person[0:t])-sum(ch_person[0:t])*CR*step/60 < CR*0.1:
			continue
		elif sum(er_person[0:t])-sum(ch_person[0:t])*CR*step/60 >= CR*0.1 and sum(er_person[0:t])-sum(ch_person[0:t])*CR*step/60 < CR*0.3 and sum(ch_person[0:t])*CR*step/60 <= sum(er_person[0:t]):
			ch_person[t] =0.1
		elif sum(er_person[0:t])-sum(ch_person[0:t])*CR*step/60 >= CR*0.1 and sum(ch_person[0:t])*CR*step/60 <= sum(er_person[0:t]):
			ch_person[t] =0.3
	base_lv2.extend(ch_person)
print('#################################################################')
print('base case: lv1 and lv2 charging (w.o penalty): ', round(ghg_cal(base_lv2),2), 'total charged energy', sum(base_lv2)*CR)
#print(feasible(base_lv2))
print(time_base(base_lv2))

base_lv3=[]
for i in range(len(person_id)):
	er_person = erij[i*len_time:(i+1)*len_time]
	ch_person = np.zeros((len(er_person),1),dtype=float).flatten().tolist()
	#	print(ch_person)
	for t in range(len(er_person)):
		if er_person[t] != 0:
			continue
		elif sum(er_person[0:t])-sum(ch_person[0:t])*CR*step/60 < CR*0.1:
			continue
		elif sum(er_person[0:t])-sum(ch_person[0:t])*CR*step/60 >= CR*0.1 and sum(er_person[0:t])-sum(ch_person[0:t])*CR*step/60 < CR*0.3 and sum(ch_person[0:t])*CR*step/60 <= sum(er_person[0:t]):
			ch_person[t] =0.1
		elif sum(er_person[0:t])-sum(ch_person[0:t])*CR*step/60 >= CR*0.3 and sum(er_person[0:t])-sum(ch_person[0:t])*CR*step/60 < CR*1 and sum(ch_person[0:t])*CR*step/60 <= sum(er_person[0:t]):
			ch_person[t] =0.3
		elif sum(er_person[0:t])-sum(ch_person[0:t])*CR*step/60 >= CR*1 and sum(ch_person[0:t])*CR*step/60 <= sum(er_person[0:t]):
			ch_person[t] =1
	base_lv3.extend(ch_person)
print('#################################################################')
print('base case: lv1, lv2, and lv3 charging (w.o penalty): ', round(ghg_cal(base_lv3),2), 'total charged energy', sum(base_lv3)*CR)
#print(feasible(base_lv3))
print(time_base(base_lv3))
