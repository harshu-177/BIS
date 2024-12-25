import numpy as np
import random
from scipy.special import gamma

def energy_function(x, y, theta):
    A = 1.5
    S = 1000
    optimal_distance = 0.0
    distance = np.sqrt(x**2 + y**2)
    energy = (A * S) / (1 + distance) * np.cos(np.radians(theta))
    return max(energy, 0)

class CuckooSearch:
    def __init__(self, fitness_function, lower_bound, upper_bound, population_size=25, max_iter=100, pa=0.25):
        self.fitness_function = fitness_function
        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)
        self.population_size = population_size
        self.max_iter = max_iter
        self.pa = pa
        self.n_dim = len(lower_bound)
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.n_dim))
        self.fitness = np.zeros(self.population_size)
        self.best_solution = None
        self.best_fitness = float('-inf')
    
    def levy_flight(self, x):
        beta = 1.5
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size=self.n_dim)
        v = np.random.normal(0, 1, size=self.n_dim)
        step = u / np.abs(v)**(1 / beta)
        new_solution = x + step * 0.01
        return np.clip(new_solution, self.lower_bound, self.upper_bound)
    
    def cuckoo_search(self):
        for iteration in range(self.max_iter):
            for i in range(self.population_size):
                new_solution = self.levy_flight(self.population[i])
                new_fitness = self.fitness_function(*new_solution)
                if new_fitness > self.fitness[i]:
                    self.population[i] = new_solution
                    self.fitness[i] = new_fitness
                if new_fitness > self.best_fitness:
                    self.best_solution = new_solution
                    self.best_fitness = new_fitness
            
            for i in range(self.population_size):
                if random.random() < self.pa:
                    self.population[i] = np.random.uniform(self.lower_bound, self.upper_bound, self.n_dim)
                    self.fitness[i] = self.fitness_function(*self.population[i])
            
            print(f"Iteration {iteration+1}/{self.max_iter}, Best Fitness (Energy): {self.best_fitness}")
        
        return self.best_solution, self.best_fitness

lower_bound = [-10, -10, 0]
upper_bound = [10, 10, 90]
population_size = 25
max_iter = 100
pa = 0.25

cuckoo_search = CuckooSearch(energy_function, lower_bound, upper_bound, population_size, max_iter, pa)
best_solution, best_fitness = cuckoo_search.cuckoo_search()

print("Optimal Solar Panel Position and Tilt Angle:")
print(f"Position (x, y): {best_solution[:2]}, Tilt Angle (theta): {best_solution[2]} degrees")
print(f"Optimal Energy (Fitness): {best_fitness} Watts")
