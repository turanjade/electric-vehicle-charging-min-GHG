import math
ef_file = open('/Users/ran/Documents/Github/charging_data/2018_gasoline_21_unres_ghg.csv', 'r')
ef = []
avg_v = []
next(ef_file)
for line in ef_file:
	col = [x.rstrip() for x in line.split(',')]
	avg_v.append(float(col[0]))
	ef.append(float(col[2]))

for i in range(10):
	ghg = 0
	trip_f = open('/Users/ran/Documents/Github/charging_data/tts5%_processed_'+str(i)+'.csv', 'r')
	next(trip_f)
	for line in trip_f:
		cols = [x.rstrip() for x in line.split(',')]
		dist_km = float(cols[7])
		avg_kmh = (float(cols[9])//5) * 5 ##round to the nearest 5kmh
		if avg_kmh == 0:
			avg_kmh = 5
		if avg_kmh > 120:
			avg_kmh = 120
		ghg = ef[avg_v.index(avg_kmh)] * dist_km + ghg
	print('sample ', str(i),' ghg=', ghg/1000)


