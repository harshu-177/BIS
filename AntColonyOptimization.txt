import random
import numpy as np

class AntColony:
    def __init__(self, dist_matrix, num_ants, num_iterations, alpha=1, beta=2, rho=0.1, q=100):
        self.dist_matrix = dist_matrix
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha 
        self.beta = beta    
        self.rho = rho   
        self.q = q         
        self.num_cities = len(dist_matrix)
        self.pheromone_matrix = np.ones((self.num_cities, self.num_cities))
        np.fill_diagonal(self.pheromone_matrix, 0)

    def select_next_city(self, current_city, visited_cities):
        probabilities = []
        for city in range(self.num_cities):
            if city not in visited_cities:
                pheromone = self.pheromone_matrix[current_city][city] ** self.alpha
                distance = self.dist_matrix[current_city][city] ** self.beta
                probabilities.append(pheromone * distance)
            else:
                probabilities.append(0)

        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]

        return np.random.choice(range(self.num_cities), p=probabilities)

    def construct_solution(self):
        visited_cities = [random.randint(0, self.num_cities - 1)]
        while len(visited_cities) < self.num_cities:
            current_city = visited_cities[-1]
            next_city = self.select_next_city(current_city, visited_cities)
            visited_cities.append(next_city)
        visited_cities.append(visited_cities[0])  
        return visited_cities

    def calculate_solution_length(self, solution):
        length = 0
        for i in range(len(solution) - 1):
            length += self.dist_matrix[solution[i]][solution[i + 1]]
        return length

    def update_pheromones(self, solutions, lengths):
        self.pheromone_matrix *= (1 - self.rho)
        for solution, length in zip(solutions, lengths):
            for i in range(len(solution) - 1):
                self.pheromone_matrix[solution[i]][solution[i + 1]] += self.q / length

    def optimize(self):
        best_solution = None
        best_length = float('inf')
        for _ in range(self.num_iterations):
            solutions = []
            lengths = []
            for _ in range(self.num_ants):
                solution = self.construct_solution()
                length = self.calculate_solution_length(solution)
                solutions.append(solution)
                lengths.append(length)
                if length < best_length:
                    best_solution = solution
                    best_length = length

            self.update_pheromones(solutions, lengths)

            print(f"Best Length in Iteration: {best_length}")

        return best_solution, best_length

def generate_distance_matrix(num_cities):
    matrix = np.random.randint(10, 100, size=(num_cities, num_cities))
    np.fill_diagonal(matrix, 0)
    return matrix


num_cities = 10
num_ants = 20
num_iterations = 100


dist_matrix = generate_distance_matrix(num_cities)


aco = AntColony(dist_matrix, num_ants, num_iterations)
best_solution, best_length = aco.optimize()

print("Best Solution (City Order):", best_solution)
print("Best Solution Length:", best_length)
