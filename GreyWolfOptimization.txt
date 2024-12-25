import numpy as np

def objective_function(x):
    return np.sum(x**2)

def grey_wolf_optimizer(obj_func, dim, bounds, max_iter, pack_size):
    alpha_pos = np.zeros(dim)
    beta_pos = np.zeros(dim)
    delta_pos = np.zeros(dim)
    alpha_score = float("inf")
    beta_score = float("inf")
    delta_score = float("inf")
    wolves = np.random.uniform(bounds[0], bounds[1], (pack_size, dim))

    for t in range(max_iter):
        for i in range(pack_size):
            fitness = obj_func(wolves[i])
            if fitness < alpha_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = alpha_score
                beta_pos = alpha_pos.copy()
                alpha_score = fitness
                alpha_pos = wolves[i].copy()
            elif fitness < beta_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = fitness
                beta_pos = wolves[i].copy()
            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = wolves[i].copy()

        a = 2 - t * (2 / max_iter)
        for i in range(pack_size):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - wolves[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - wolves[i, j])
                X2 = beta_pos[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - wolves[i, j])
                X3 = delta_pos[j] - A3 * D_delta

                wolves[i, j] = (X1 + X2 + X3) / 3
            wolves[i] = np.clip(wolves[i], bounds[0], bounds[1])

    return alpha_pos, alpha_score

dim = 5
bounds = (-10, 10)
max_iter = 100
pack_size = 20

best_position, best_score = grey_wolf_optimizer(objective_function, dim, bounds, max_iter, pack_size)

print("Best Position:", best_position)
print("Best Score:", best_score)
