
import numpy as np

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
        for i, x_i in enumerate(data):
            euclidean_norms = [np.linalg.norm(x_i - self.mu[j])  for j in range(self.n_comp)]
            denom = 1 / k * np.power(np.sum([np.power(eu_norm, 2 * s) for eu_norm in euclidean_norms]), 1 - 1 / s)
            weights[i] = 1 / k * np.array([np.power(eu_norm, 2 * (s - 1)) for eu_norm in euclidean_norms]) / denom
        return weights
    
    def calculate_centers(self, data, weights):
        weights_mul_x = weights.T @ data
        denom = np.sum(weights, axis=0)
        self.mu = weights_mul_x / np.expand_dims(denom, 1)
        return self.mu
        
    def calculate_obj(self, data, mu=None):
        if mu is None:
            mu = self.mu
        return np.array([[np.linalg.norm(data[i] - mu[j]) ** 2 for i in range(data.shape[0])] for j in range(self.n_comp)]).min(axis=0).sum()
        
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


import pandas as pd
from scipy.spatial import distance_matrix

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
            pkm = PowerKM(init_centers, s_0=-1)
            pkm.fit(data)
            centroids = pkm.mu
            print(f'PKM: {inertia(centroids, data)}')
            if inertia(centroids, data) < best_inertia:
                best_inertia = inertia(centroids, data)
            
        # centroids = pkm.mu
        results.append(best_inertia)
    
    print(f'Res: {np.mean(results)} +- {np.std(results)}')

    print(init_km.inertia_)
    pkm = PowerKM(init_centers)
    
    print(f'Just KM: {inertia(init_centers, data)}')
    print(f'Just KM: {pkm.calculate_obj(data, mu=init_centers)}')
    pkm.fit(data)
    pkm_centers = pkm.mu
    # emply_km = KMeans(n_comp, init=pkm_centers, max_iter=0)
    # emply_km.fit(data)
    # print(emply_km.score(data))
