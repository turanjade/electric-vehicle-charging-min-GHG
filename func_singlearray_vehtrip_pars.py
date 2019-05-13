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

def convert_array(person_id, tts_sample, col_to_convert_orig, col_to_convert_dest, step, col_personid, col_starttime, col_endtime):
	colnum_sz = col_to_convert_orig
	colnum_ez = col_to_convert_dest
	colnum_p = col_personid
	colnum_st = col_starttime
	colnum_et = col_endtime
	
	zonetype_min = []
	for i in range(len(person_id)):
		indices = np.where(tts_sample[:,colnum_p]==person_id[i])
		persontrips = tts_sample[indices[0],:]
		zonetype_min_person = []
		tp = 0 ##time point
		
		#####initialize, at the beginning of the tour (start of a day)
		while tp < int(persontrips[0,colnum_st]):
			zonetype_min_person.append(persontrips[0,colnum_sz])
			tp += 1
		
		#####if only one trip per day
		if len(indices[0]) == 1:
			xxxx = 'only one trip'
			while tp < 1440:
			###during the rest of the time, zone is the destination zone of the single trip
				zonetype_min_person.append(persontrips[0,colnum_ez])
				tp+=1
			#print(i, person_id[i], tp, 'only one trip one day')
			zonetype_min.extend(zonetype_min_person)
			if tp != 1440:
				print(i, person_id[i], tp, 'only one trip one day, not complete')
			continue
		
		#####multiple trips
		else:
			for t in range(1,len(indices[0])):
				while tp < int(persontrips[t,colnum_st]):
					zonetype_min_person.append(persontrips[t,colnum_sz])
					tp+=1
			if float(persontrips[t,colnum_et]) < 1440:
				while tp < 1440:
					zonetype_min_person.append(persontrips[t,colnum_ez])
					tp+=1
			elif float(persontrips[t,colnum_et]) >= 1440:
				while tp < 1440:
					zonetype_min_person.append(persontrips[t,colnum_ez])
					tp+=1
			#print(i, person_id[i], tp, 'multiple trips per day')
		zonetype_min.extend(zonetype_min_person)
		if tp != 1440:
			print(i, person_id[i], tp, 'multiple trips one day, not complete, endtime', persontrips[t,colnum_et])

	print('zonetype_min finish', len(zonetype_min))
	
	##convert zone per min to zone per step
	count = 0
	zonetype = []
	zone_ini = []
	
	for i in range(len(zonetype_min)):
		#for type in zonetype_min:
		#count = i+1
		if (i+1)%step != 0: #######here step can be changed to any value, append all possible zone types
			zone_ini.append(zonetype_min[i])
		elif (i+1)%step == 0:
			zone_ini.append(zonetype_min[i])
			if len(np.unique(zone_ini)) == 1 and zone_ini[0] == 'H':
				zonetype.append('H')
			else:
				zonetype.append('O')
			zone_ini = []

#		else:
#			zone_ini_uniq = np.unique(zone_ini)
#			zone_ini_uniq_len = []
#			for j in zone_ini_uniq:
#				zone_ini_uniq_len.append(len(np.argwhere(np.array(zone_ini)==j).flatten().tolist()))
#		zonetype.append(zone_ini_uniq[np.argwhere(np.array(zone_ini_uniq_len)==max(zone_ini_uniq_len)).flatten().tolist()[0]])
#			zone_ini=[]

	return zonetype

