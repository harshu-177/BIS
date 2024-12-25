import numpy as np

def rastrigin_function(x, y):
    return 10 * 2 + (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y))

def initialize_grid(grid_size, bounds):
    return np.random.uniform(bounds[0], bounds[1], (grid_size, grid_size, 2))

def evaluate_grid(grid, fitness_function):
    fitness_grid = np.zeros((grid.shape[0], grid.shape[1]))
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            x, y = grid[i, j]
            fitness_grid[i, j] = fitness_function(x, y)
    return fitness_grid

def get_neighbors(grid, i, j):
    neighbors = [
        grid[(i - 1) % grid.shape[0], j],
        grid[(i + 1) % grid.shape[0], j],
        grid[i, (j - 1) % grid.shape[1]],
        grid[i, (j + 1) % grid.shape[1]],
    ]
    return neighbors

def update_grid(grid, fitness_grid, bounds):
    new_grid = grid.copy()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            neighbors = get_neighbors(grid, i, j)
            best_neighbor = min(neighbors, key=lambda n: rastrigin_function(n[0], n[1]))
            if rastrigin_function(best_neighbor[0], best_neighbor[1]) < fitness_grid[i, j]:
                new_grid[i, j] = best_neighbor + np.random.uniform(-0.1, 0.1, size=2)
                new_grid[i, j] = np.clip(new_grid[i, j], bounds[0], bounds[1])
    return new_grid

def parallel_cellular_algorithm(fitness_function, grid_size, bounds, max_iter):
    grid = initialize_grid(grid_size, bounds)
    for _ in range(max_iter):
        fitness_grid = evaluate_grid(grid, fitness_function)
        grid = update_grid(grid, fitness_grid, bounds)
    best_cell = min(grid.reshape(-1, 2), key=lambda c: fitness_function(c[0], c[1]))
    best_fitness = fitness_function(best_cell[0], best_cell[1])
    return best_cell, best_fitness

grid_size = 10
bounds = (-5.12, 5.12)
max_iter = 100

best_solution, best_fitness = parallel_cellular_algorithm(rastrigin_function, grid_size, bounds, max_iter)

print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
