import numpy as np
import math

#input description: personid list, tts raw data, charging profile, orig zone id column, dest zone id column, profile update step, person id column, starttime column, endtime column
def zone_cs_count(ch_profile, zoneid_sample, step):
	
	zoneid_sample = zoneid_sample#[0:48]
	ch_profile = ch_profile#[0:48]
	#print('zone profile', zoneid_sample)
	#print('charging profile', ch_profile)
	cs_count_lv1 = []
	cs_count_lv2 = []
	cs_count_lv3 = []
	zoneid_list = []
	zone_id = np.unique(zoneid_sample)
	#print(len(zone_id))
	#print('unique zoneid=', zone_id)
	
	for i in range(len(zone_id)):
		zone_i_lv1 = [0 for k in range(1440/step)]
		zone_i_lv2 = [0 for k in range(1440/step)]
		zone_i_lv3 = [0 for k in range(1440/step)]
		
		#list all occurrences and their indices of zone id in the single-array zoneid profile
		zone_indices = [x for x, k in enumerate(zoneid_sample) if k == zone_id[i]]
		#list charging status corresponding to zone indices
		ch_state = [ch_profile[x] for x in zone_indices]
		
		##if this zone does not show any charging activity in the charging profile
		if len(ch_state) == 0:
			print('this zone does not exist', zone_id[i])
		else:
			for j in range(len(ch_state)):
				if ch_state[j] == 1.6:
					#print('zone indices for lv1=', zone_indices[j])
					zone_i_lv1[zone_indices[j] % 24] += 1
				elif ch_state[j] == 7:
					#print('zone indices for lv2=', zone_indices[j])
					zone_i_lv2[zone_indices[j] % 24] += 1
				elif ch_state[j] == 50:
					#print('zone indices for lv3=', zone_indices[j])
					zone_i_lv3[zone_indices[j] % 24] += 1
			zoneid_list.append(zone_id[i])
			cs_count_lv1.append(zone_i_lv1)
			cs_count_lv2.append(zone_i_lv2)
			cs_count_lv3.append(zone_i_lv3)
				
	#for i in range(len(ch_profile)):
	#	zone_i_lv1 = [0 for i in range(1440/step)]
	#	zone_i_lv2 = [0 for i in range(1440/step)]
	#	zone_i_lv3 = [0 for i in range(1440/step)]
	#	for j in range(len(zone_id)):
	#		if zonetype[i] == zone_id[j] and ch_profile[i] == float(1.6):
	#			zone_i_lv1[math.ceil(i/(1440/step))] += 1
	#		elif zonetype[i] == zone_id[j] and ch_profile[i] == float(7):
	#			zone_i_lv2[math.ceil(i/(1440/step))] += 1
	#		elif zonetype[i] == zone_id[j] and ch_profile[i] == float(50):
	#			zone_i_lv1[math.ceil(i/(1440/step))] += 1
	#	cs_count_lv1.append(zone_i_lv1)
	#	cs_count_lv2.append(zone_i_lv2)
	#	cs_count_lv3.append(zone_i_lv3)

	cs_count_lv1_max = [max(cs_count_lv1[i]) for i in range(len(cs_count_lv1))]
	cs_count_lv2_max = [max(cs_count_lv2[i]) for i in range(len(cs_count_lv2))]
	cs_count_lv3_max = [max(cs_count_lv3[i]) for i in range(len(cs_count_lv3))]

	return	zoneid_list, cs_count_lv1_max, cs_count_lv2_max, cs_count_lv3_max

