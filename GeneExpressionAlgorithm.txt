import numpy as np

def sphere_function(x):
    return np.sum(x**2)

def initialize_population(pop_size, gene_length, bounds):
    return np.random.uniform(bounds[0], bounds[1], (pop_size, gene_length))

def evaluate_fitness(population, fitness_function):
    return np.array([fitness_function(individual) for individual in population])

def select_parents(population, fitness, num_parents):
    sorted_indices = np.argsort(fitness)
    return population[sorted_indices[:num_parents]]

def crossover(parents, offspring_size):
    offspring = np.zeros(offspring_size)
    for i in range(offspring_size[0]):
        p1, p2 = np.random.choice(parents.shape[0], 2, replace=False)
        crossover_point = np.random.randint(1, offspring_size[1])
        offspring[i, :crossover_point] = parents[p1, :crossover_point]
        offspring[i, crossover_point:] = parents[p2, crossover_point:]
    return offspring

def mutate(offspring, mutation_rate, bounds):
    for i in range(offspring.shape[0]):
        for j in range(offspring.shape[1]):
            if np.random.rand() < mutation_rate:
                offspring[i, j] += np.random.uniform(-0.1, 0.1)
                offspring[i, j] = np.clip(offspring[i, j], bounds[0], bounds[1])
    return offspring

def gene_expression_algorithm(fitness_function, pop_size, gene_length, bounds, num_generations, mutation_rate, num_parents):
    population = initialize_population(pop_size, gene_length, bounds)
    best_solution, best_fitness = None, float("inf")
    for _ in range(num_generations):
        fitness = evaluate_fitness(population, fitness_function)
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_solution, best_fitness = population[best_idx], fitness[best_idx]
        parents = select_parents(population, fitness, num_parents)
        offspring_size = (pop_size - num_parents, gene_length)
        offspring = crossover(parents, offspring_size)
        offspring = mutate(offspring, mutation_rate, bounds)
        population[:num_parents] = parents
        population[num_parents:] = offspring
    return best_solution, best_fitness

pop_size = 50
gene_length = 5
bounds = (-5.12, 5.12)
num_generations = 100
mutation_rate = 0.1
num_parents = 10

best_solution, best_fitness = gene_expression_algorithm(sphere_function, pop_size, gene_length, bounds, num_generations, mutation_rate, num_parents)

print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
