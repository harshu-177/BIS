import random
import numpy as np

class Particle:
    def __init__(self, dim, bounds):
        self.position = np.array([random.uniform(bounds[0], bounds[1]) for _ in range(dim)])
        self.velocity = np.array([random.uniform(-1, 1) for _ in range(dim)])
        self.best_position = self.position.copy()
        self.best_score = float('inf')

    def update_velocity(self, global_best_position, w, c1, c2):
        inertia = w * self.velocity
        cognitive = c1 * random.random() * (self.best_position - self.position)
        social = c2 * random.random() * (global_best_position - self.position)
        self.velocity = inertia + cognitive + social

    def update_position(self, bounds):
        self.position = self.position + self.velocity
        self.position = np.clip(self.position, bounds[0], bounds[1])

    def evaluate(self, objective_function):
        score = objective_function(self.position)
        if score < self.best_score:
            self.best_score = score
            self.best_position = self.position.copy()


class PSO:
    def __init__(self, objective_function, dim, bounds, num_particles, max_iter, w=0.5, c1=1.5, c2=1.5):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.particles = [Particle(dim, bounds) for _ in range(num_particles)]

    def optimize(self):
        for iteration in range(self.max_iter):
            for particle in self.particles:
                particle.evaluate(self.objective_function)
                if particle.best_score < self.global_best_score:
                    self.global_best_score = particle.best_score
                    self.global_best_position = particle.best_position.copy()

            for particle in self.particles:
                particle.update_velocity(self.global_best_position, self.w, self.c1, self.c2)
                particle.update_position(self.bounds)

            print(f"Iteration {iteration+1}/{self.max_iter}, Best Score: {self.global_best_score}")

        return self.global_best_position, self.global_best_score


def objective_function(x):
    return sum(xi**2 for xi in x)

dim = 2
bounds = (-10, 10)
num_particles = 30
max_iter = 100

pso = PSO(objective_function, dim, bounds, num_particles, max_iter)
best_position, best_score = pso.optimize()

print("Optimal Solution: ", best_position)
print("Best Score: ", best_score)
