from func_singlearray_zoneidwrite import convert_array
inputdir = '/Users/ran/Documents/Github/charging_data'
outputdir = '/Users/ran/Documents/Github/charging_results/Sample_Timebased_Zoneid'
for i in range(10):
	convert_array(inputdir, outputdir, str(i), 60)

