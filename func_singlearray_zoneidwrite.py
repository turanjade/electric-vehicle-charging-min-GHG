import numpy as np
import math
import functools

#input description: personid list, tts raw data, charging profile, orig zone id column, dest zone id column, profile update step, person id column, starttime column, endtime column
def convert_array(inputdir, outputdir, num_sample, step):
	
	tts_sample = []
	#tts_sample_file = open('tts5%_processed_0.csv', 'r')
	tts_sample_file = open(inputdir+'/tts5%_processed_'+num_sample+'.csv', 'r')
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
	col_origid = colnames.index('gta06_orig')
	col_destid = colnames.index('gta06_dest')

	#########var settings
	flexible = 1 ####setup a turn on/off button to switch between two mef modes
	person_id = np.unique(np.array(tts_sample[:,col_personid]))#[0:4] ##travelers id, aka veh id
		#print('person_id', person_id)
		
	colnum_sz = col_origid
	colnum_ez = col_destid
	colnum_p = col_personid
	colnum_st = col_starttime
	colnum_et = col_endtime
	
	len_time_1min = 1440  ###total min of a day

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
	
	##convert zone id per min to zone per step
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
			if len(np.unique(zone_ini)) == 1: #means this veh stays in the same place the whole time and zone_ini[0] == 'H':
				zonetype.append(zone_ini[0])
			else:
				zonetype.append(9999999) #9999999 (seven nine's: this veh is on road)
			zone_ini = []

	zonetype_f = open(outputdir+'/zoneid_'+num_sample+'.csv', 'w')
	for i in zonetype:
		zonetype_f.write(str(i)+'\n')
	zonetype_f.close()

	return	0
