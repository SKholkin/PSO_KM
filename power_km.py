
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix


class PowerKM:
    def __init__(self, init_centers, s_0=-5) -> None:
        self.eta = 1.1
        self.s_0 = s_0
        self.mu = init_centers
        self.n_comp, self.dim = init_centers.shape[0], init_centers.shape[1]
        self.weights = None
        self.annealing_iters = 50
        
    def calculate_weights(self, data, s):
        k = self.n_comp
        weights = np.zeros([data.shape[0], self.n_comp])
        
        dist_matr = distance_matrix(self.mu, data, p=2)

        denom = 1 / k * np.power(np.sum(np.power(dist_matr, 2 * s), axis=0), 1 - 1 / s)

        weights = np.divide(1 / k * np.power(dist_matr, 2 * (s - 1)), denom).T
        return weights
    
    def calculate_centers(self, data, weights):
        weights_mul_x = weights.T @ data
        denom = np.sum(weights, axis=0)
        self.mu = weights_mul_x / np.expand_dims(denom, 1)
        return self.mu
        
    def calculate_obj(self, data, mu=None):
        if mu is None:
            mu = self.mu
        
        dist_matr = distance_matrix(mu, data, p=2)
        return dist_matr.min(axis=0).sum()
        
    def fit(self, data):
        s = self.s_0
        self.weights = np.zeros([self.n_comp, data.shape[0]])
        for m in range(self.annealing_iters):
            weights = self.calculate_weights(data, s)
            prev_mu = self.mu
            self.mu = self.calculate_centers(data, weights)
            # print(f'S: {s} objective: {self.calculate_obj(data)}')
            if np.isnan(self.calculate_obj(data)):
                self.mu = prev_mu
                self.finetune()
                break
            s = self.eta * s
        self.finetune()
    
    def finetune(self):
        # print(f'Before KM {self.calculate_obj(data)}')
        km = KMeans(self.mu.shape[0], init=self.mu, n_init=1)
        km.fit(data)
        self.mu = km.cluster_centers_
        # print(f'After KM {self.calculate_obj(data)}')


def kmeanspp(data, n_comp):
    rand_idx = np.random.randint(0, len(data))
    centroids = np.zeros([n_comp, data.shape[1]])
    centroids[0] = data[rand_idx]
    for i in range(1, n_comp):
        curr_centroids = centroids[:i]
        dist_matr = distance_matrix(curr_centroids, data, p=2)
        min_data_dist = np.min(dist_matr, axis=0)
        x = np.multiply((dist_matr <= np.repeat(np.expand_dims(min_data_dist, axis=0), dist_matr.shape[0], axis=0)), dist_matr)
        max_centroids_dist_idx = np.argmax(np.max(x, axis=0), axis=0)
        centroids[i] = data[max_centroids_dist_idx]

    return centroids

if __name__ == '__main__':
    from sklearn.cluster import KMeans
    from utils import load_cloud_dataset, load_texture_dataset, inertia
    from time import time

    np.random.seed(0)
    data = load_texture_dataset()
    N = 2
    n_comp = 11
    n_runs = 10
    results = []
    for i in range(n_runs):
        best_inertia = np.inf
        for j in range(N):
            init_km = KMeans(n_comp, n_init=1)
            init_km.fit(data)
            print(f'KM: {init_km.score(data)}')
            init_centers = init_km.cluster_centers_
            start = time()
            pkm = PowerKM(init_centers, s_0=-1)
            pkm.fit(data)
            print(f'PKM fitting time: {time() - start}')
            centroids = pkm.mu
            print(f'PKM: {inertia(centroids, data)}')
            if inertia(centroids, data) < best_inertia:
                best_inertia = inertia(centroids, data)
            
        # centroids = pkm.mu
        results.append(best_inertia)
    
    print(f'Res: {np.mean(results)} +- {np.std(results)}')
