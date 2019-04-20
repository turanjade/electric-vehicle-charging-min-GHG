from func_tripselect import ttsselect_5percent

tts_sample_file = open('tts_toydata.csv','r')

colnames, tts_sample = ttsselect_5percent(tts_sample_file)

tts_toy_processed = open('tts_toydataselect1.csv', 'w')
tts_toy_processed.write(','.join(str(i) for i in colnames)+'\n')
for trip in tts_sample:
	str_data = ','.join(str(i) for i in trip)
	tts_toy_processed.write(str_data+'\n')