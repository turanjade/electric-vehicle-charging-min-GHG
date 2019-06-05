####should be copy/paste to other code files and run

population = toolbox.population(n=5)
print(population)
print(type(population))
fitnesses = toolbox.map(toolbox.evaluate, population)
for ind, fit in zip(population, fitnesses):
	ind.fitness.values = fit
#	print(fit)
#print(population[0].fitness.values[0])

population_val=tuple([])
parent_val = population_val
for p in range(0, len(population)):
	population_val = population_val + (population[p].fitness.values)
population = [x for _, x in sorted(zip(population_val, population))]
print(population)
print(population[0].fitness.values)
print(population[1].fitness.values)
print(population[2].fitness.values)
print(population[3].fitness.values)
print(population[4].fitness.values)
print(parent_val)
parent_val = population_val
print(parent_val)
#print(ind)
exit()