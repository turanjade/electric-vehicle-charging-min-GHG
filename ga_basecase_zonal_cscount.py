from func_zone_cscount import zone_cs_count

optimized = 0
bc_home = 0
bc_nohome = 0
bc_aftertrip = 0
bc_after3am = 1

for s in range(10):
	num_sample = str(s)
	####write lv123 cs count file, and read plan file
	if optimized == 1:
		lv123_f = open('/Users/ran/Documents/Github/charging_results/SA_TTS5%_1hour_ChargingStation/Opt_lv123_max_cs_eachzone_'+num_sample+'.csv','w')
		plan_file = open('/Users/ran/Documents/Github/charging_results/SA_BaseCase_TTS5%_1hour/Full_newlevel_TTS5%_sample'+num_sample+'_SA_cx0.7mut0.05_ngen30lambda100.txt','r')
	
	elif bc_home == 1:
		lv123_f = open('/Users/ran/Documents/Github/charging_results/SA_BaseCase_TTS5%_1hour_ChargingStation/BC_home_lv123_max_cs_eachzone_'+num_sample+'.csv','w')
		plan_file = open('/Users/ran/Documents/Github/charging_results/Basecase_TTS5%_1hour/BC_home_'+num_sample+'_lv3.csv','r')
	
	elif bc_nohome == 1:
		lv123_f = open('/Users/ran/Documents/Github/charging_results/SA_BaseCase_TTS5%_1hour_ChargingStation/BC_nohome_lv123_max_cs_eachzone_'+num_sample+'.csv','w')
		plan_file = open('/Users/ran/Documents/Github/charging_results/Basecase_TTS5%_1hour/BC_nohome_'+num_sample+'_lv3.csv','r')
	
	elif bc_aftertrip == 1:
		lv123_f = open('/Users/ran/Documents/Github/charging_results/SA_BaseCase_TTS5%_1hour_ChargingStation/BC_aftertrip_lv123_max_cs_eachzone_'+num_sample+'.csv','w')
		plan_file = open('/Users/ran/Documents/Github/charging_results/Basecase_TTS5%_1hour/BC_aftertrip_'+num_sample+'_lv3.csv','r')
	
	elif bc_after3am == 1:
		lv123_f = open('/Users/ran/Documents/Github/charging_results/SA_BaseCase_TTS5%_1hour_ChargingStation/BC_after3am_lv123_max_cs_eachzone_'+num_sample+'.csv','w')
		plan_file = open('/Users/ran/Documents/Github/charging_results/Basecase_TTS5%_1hour/BC_after3am_'+num_sample+'_lv3.csv','r')

#######################################################evaluation##########################################
	plan = []
	for x in plan_file:
		plan.append(float(x.rstrip()))
	plan_file.close()

	zoneid_sample = []
	zoneid_sample_file = open('/Users/ran/Documents/Github/charging_results/Sample_Timebased_Zoneid/zoneid_'+num_sample+'.csv', 'r')
	for x in zoneid_sample_file:
		zoneid_sample.append(float(x.rstrip()))
	zoneid_sample_file.close()

	cs_list = zone_cs_count(plan, zoneid_sample, 60)

	zoneid = cs_list[0]
	lv1_cs = cs_list[1]
	lv2_cs = cs_list[2]
	lv3_cs = cs_list[3]
	print(len(zoneid), len(lv1_cs), len(lv2_cs), len(lv3_cs))
	for i in range(len(zoneid)):
		lv123_f.write(str(zoneid[i])+','+str(lv1_cs[i])+','+str(lv2_cs[i])+','+str(lv3_cs[i])+'\n')
	lv123_f.close()

