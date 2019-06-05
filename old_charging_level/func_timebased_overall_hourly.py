#######define functions to analyze charging results

#######generate time-based charging result - over all GTA

def time_base(charge_profile):
	num_person = len(charge_profile)/24
	time_charge = []
	for i in range(24):
		time_index = [p*24+i for p in range(num_person)]
		time_charge.append(sum([charge_profile[p] for p in time_index])*50)
	return(time_charge)

#plan_file = open('/Users/ran/Documents/Github/charging_results/SA_TTS5%_1hour/Full_TTS5%_sample2_SA_cx0.7mut0.05_ngen30lambda100_randomseeds1813.txt','r')##29205
#plan = []
#for x in plan_file:
#	plan.append(float(x.rstrip()))
#plan_file.close()
#print(time_base(plan))
#print(sum(time_base(plan)))
#print(sum(plan)*50)
