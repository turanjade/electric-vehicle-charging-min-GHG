from func_result_evaluate_0 import min_ghg_MEF, feasible, check_feasible

plan_file = open('/Users/ran/Documents/Github/charging_results/SA_TTS5%_1hour/Full_TTS5%_sample0_SA_cx0.7mut0.05_ngen30lambda100_randomseeds76985.txt','r')
plan = []
for x in plan_file:
	plan.append(float(x.rstrip()))
plan_file.close()
if feasible(plan) is False:
	print('solution infeasible')
	print(check_feasible(plan))
print(min_ghg_MEF(plan))
#	print('randomseed', str(i), ' fixed ghg=', fixed.ghg_cal(plan))
#	print('randomseed', str(i), ' flexible ghg=', flexible.ghg_cal(plan))
