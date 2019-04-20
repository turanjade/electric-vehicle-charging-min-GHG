from func_tripselect import ttsselect_5percent

tts_full_file = open('ttsfull_processed.csv','r')

for i in range(0,10):
	colnames, tts_sample = ttsselect_5percent(tts_full_file)
	tts_sample_file = open('tts5%_processed_' + str(i) + '.csv', 'w')
	tts_sample_file.write(','.join(text for text in colnames)+ '\n')
	for trips in tts_sample:
		tts_sample_file.write(','.join(str(j) for j in trips)+ '\n')
	tts_sample_file.close()