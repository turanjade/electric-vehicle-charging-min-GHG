import random

def ttsselect_5percent(ttsfile):
	##########select 5% tours (vehicles) as EV first
	##########then for each vehicle, if the stop duration is smaller than 2 hour, then merge these two trip to one
	##########if the total energy consumption is smaller than 1 KWh, delete the entire tours
	i = 0
	
	personidlist = []
	person_to_delete = []
	personcount = -1
	personenergy = []
	sampletrip_1 = [] ########this stores the trip with duration longer than 2 hour
	for line in ttsfile:
		if i == 0:
			cols = [x.rstrip() for x in line.split(',')]
			colnames = cols
			print(colnames)
			i += 1
			
			person_id = colnames.index('person_id')
			energy = colnames.index('energy')
			starttime = colnames.index('starttime')
			endtime = colnames.index('endtime')
			orig = colnames.index('gta06_orig')
			dest = colnames.index('gta06_dest')
			purporig = colnames.index('purp_orig')
			purpdest = colnames.index('purp_dest')
			dist = colnames.index('dist')
			time = colnames.index('time')
			num = colnames.index('trip_num')

			continue
			
		cols = [x.rstrip() for x in line.split(',')]
		randnum = random.random()
		if cols[person_id] not in personidlist:# and randnum <= 0.05:
			
			if len(personenergy)>0 and float(personenergy[personcount-1]) <= 1:
				person_to_delete.append(personidlist[personcount-1]) #####record tours with energy total less than 1KWh
			#	print(personidlist[personcount-1], personenergy[personcount-1])
			personcount += 1
			personidlist.append(cols[person_id])

			personenergy.append(float(cols[energy]))
			
			tripnum = cols[num]
			tripopurp = cols[purporig]
			tripo = cols[orig]
			tripdpurp = cols[purpdest]
			tripd = cols[dest]
			tripdist = float(cols[dist])
			triptime = float(cols[time])
			tripenergy = float(cols[energy])
			tripstart = cols[starttime]
			tripend = cols[endtime]
			tripspeed = tripdist/(triptime/60)
			
			tripnum = 1
			
			sampletrip_1.append([tripnum, tripopurp, tripo, tripdpurp, tripd, 'D', cols[person_id], tripdist, triptime, tripspeed, tripenergy, tripstart, tripend])
			
		elif cols[person_id] in personidlist:
			personenergy[personcount] = personenergy[personcount] + float(cols[energy])
			
			tripnum_new = cols[num]
			tripopurp_new = cols[purporig]
			tripo_new = cols[orig]
			tripdpurp_new = cols[purpdest]
			tripd_new = cols[dest]
			tripdist_new = float(cols[dist])
			triptime_new = float(cols[time])
			tripenergy_new = float(cols[energy])
			tripstart_new = cols[starttime]
			tripend_new = cols[endtime]
				
			if float(tripstart_new) - float(tripend) > 120:
				tripspeed = tripdist_new/(triptime_new/60)
				sampletrip_1.append([tripnum_new, tripopurp_new, tripo_new, tripdpurp_new, tripd_new, 'D', cols[person_id], tripdist_new, triptime_new, tripspeed, tripenergy_new, tripstart_new, tripend_new])
				tripnum = tripnum_new
				tripopurp = tripopurp_new
				tripo = tripo_new
				tripdpurp = tripdpurp_new
				tripd = tripd_new
				tripdist = tripdist_new
				triptime = triptime_new
				tripenergy = tripenergy_new
				tripstart = tripstart_new
				tripend = tripend_new
				
			else:
				sampletrip_1 = sampletrip_1[0:-1]
				tripd = tripd_new
				tripdpurp = tripdpurp_new
				tripdist = tripdist + tripdist_new
				triptime = triptime + triptime_new
				tripspeed = tripdist/(triptime/60)
				tripenergy = tripenergy + tripenergy_new
				tripstart = tripstart 
				tripend = tripend_new
				
				sampletrip_1.append([tripnum, tripopurp, tripo, tripdpurp, tripd, 'D', cols[person_id], tripdist, triptime, tripspeed, tripenergy, tripstart, tripend])
	
	############then delete tours with total energy less than 1
	sampletrip_2 = []
	colnames = ['trip_num','purp_orig','gta06_orig', 'purp_dest', 'gta06_dest', 'mode_prime', 'person_id', 'dist', 'time', 'kmh', 'energy', 'starttime', 'endtime']
	for trip in sampletrip_1:
		if trip[6] not in person_to_delete:
			sampletrip_2.append(trip)
	return colnames, sampletrip_2