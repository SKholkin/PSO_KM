from sklearn.preprocessing import MinMaxScaler
import os.path as osp
import pandas as pd
import numpy as np
from math import copysign, hypot
from sklearn.datasets import load_digits

real_dataset_names = ['breast_cancer', 'cloud']

def load_texture_dataset():
    file_name = 'data/texture.dat'
    with open(file_name) as f:
        texture_data = pd.DataFrame([item.split(',') for item in f.readlines()])
    texture_data = texture_data.astype(float).to_numpy()[:, :-1]
    print(texture_data.shape)

    x_min, x_max = np.min(texture_data, axis=0), np.max(texture_data, axis=0)

    texture_data = (texture_data - x_min) / (x_max - x_min)
    scaler = MinMaxScaler()

    # scaler.fit(texture_data)
    # texture_data = scaler.transform(texture_data)
    return texture_data

def load_breast_cancer():
    bc_data_name = 'data/wdbc.data'
    with open(bc_data_name) as f:
        data_bc = pd.DataFrame([item.split(',')[2:] for item in f.readlines()])
    data_bc = data_bc.astype(float).to_numpy()
    scaler = MinMaxScaler()
    scaler.fit(data_bc)
    data_bc = scaler.transform(data_bc)
    return data_bc

def load_cloud_dataset():
    cloud_data_name = 'data/cloud.data'
    with open(cloud_data_name) as f:
        cloud_data = pd.DataFrame([item.split() for item in f.readlines()])
    cloud_data = cloud_data.astype(float).to_numpy()
    scaler = MinMaxScaler()
    scaler.fit(cloud_data)
    cloud_data = scaler.transform(cloud_data)
    return cloud_data

def load_satelite_dataset():
    f_name = osp.join('data', 'sat.trn')
    with open(f_name) as f:
        sat_data = pd.DataFrame([item.split()[:-1] for item in f.readlines()])
    sat_data = sat_data.astype(float).to_numpy()
    scaler = MinMaxScaler()
    scaler.fit(sat_data)
    sat_data = scaler.transform(sat_data)
    return sat_data[:int(sat_data.shape[0] / 4), :]

def load_seg_data():
    f_name = osp.join('data', 'segmentation.data')
    with open(f_name) as f:
        sat_data = pd.DataFrame([item.split(',')[1:] for item in f.readlines()])
    sat_data = sat_data[6:].astype(float).to_numpy()
    scaler = MinMaxScaler()
    scaler.fit(sat_data)
    sat_data = scaler.transform(sat_data)
    return sat_data

def load_synthetic_dataset(filename):
    data = np.load(filename)
    scaler = MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    return data

def load_usps_dataset():
    import h5py
    filename = 'data/usps.h5'
    with h5py.File(filename, 'r') as hf:
            train = hf.get('train')
            X_tr = train.get('data')[:]
            y_tr = train.get('target')[:]
            test = hf.get('test')
            X_te = test.get('data')[:]
            y_te = test.get('target')[:]
    print(np.array(X_tr).shape)

def load_digits_dataset():
    return load_digits().data

def load_dataset(dataset_name):
    if dataset_name == 'breast_cancer':
        return load_breast_cancer()
    if dataset_name == 'cloud':
        return load_cloud_dataset()
    if dataset_name == 'landsat':
        return load_satelite_dataset()
    if dataset_name == 'seg':
        return load_seg_data()
    if dataset_name == 'digits':
        return load_digits_dataset()
    if osp.isfile(dataset_name):
        return load_synthetic_dataset(dataset_name)
    raise RuntimeError(f'Unknown dataset {dataset_name}. Please provide path to synthetic dataset file or correctly write real dataset name')

def _givens_rotation_matrix_entries(a, b):
    """Compute matrix entries for Givens rotation.[[cos(phi), -sin(phi)], [sin(phi), cos(phi)]]"""
    r = hypot(a, b)
    if r < 1e-05:
        return (1, 0)
    c = a/r
    s = -b/r

    return (c, s)

def QRGivens(A):
    if np.linalg.det(A) < 0:
        A = -A
    """Perform QR decomposition of matrix A using Givens rotation."""
    (num_rows, num_cols) = np.shape(A)

    # Initialize orthogonal matrix Q and upper triangular matrix R.
    Q = np.identity(num_rows)
    R = np.copy(A)
    phi_list = []

    # Iterate over lower triangular matrix.
    (rows, cols) = np.tril_indices(num_rows, -1, num_cols)
    i = 0
    for (row, col) in zip(rows, cols):
        i += 1

        # Compute Givens rotation matrix and
        # zero-out lower triangular matrix entries.
        (c, s) = _givens_rotation_matrix_entries(R[col, col], R[row, col])

        phi = np.arccos(c)
        if s > 0:
            phi = -phi

        if c * R[col, col] - s * R[row, col] < 0:
            phi = phi - np.pi
            c = -c
            s = -s
        phi_list.append(phi)

        R[col], R[row] = R[col]*c + R[row]*(-s), R[col]*s + R[row]*c
        Q[:, col], Q[:, row] = Q[:, col]*c + Q[:, row]*(-s), Q[:, col]*s + Q[:, row]*c
    return np.array(phi_list)


def Givens2Matrix(phi_list):
    d  = int((1 + np.sqrt(1 + 8 * len(phi_list))) / 2)
    ret_val = np.eye(d)
    i = 0
    (rows, cols) = np.tril_indices(d, -1, d)
    for (row, col) in zip(rows, cols):
        
        c = np.cos(phi_list[i])
        s = -np.sin(phi_list[i])
        i += 1

        G = np.eye(d)
        G[[col, row], [col, row]] = c

        G[row, col] = s
        G[col, row] = -s
        
        ret_val = np.dot(ret_val, G.T)
        
    return ret_val
        
def eigh_with_fixed_direction_range(spd_matr):
    eigenvalues, v = np.linalg.eigh(spd_matr)

    base_vector = np.ones_like(v[0])
    for i in range(v.shape[0]):
        cos_phi = np.dot(base_vector, v[:, i])
        if cos_phi > 0:
            v[:, i] = -v[:, i]

    return eigenvalues, v

def find_closest_spd(A):
    eps = 1e-05
    w, v  = np.linalg.eigh(A)
    w = w * (w > 0) + eps
    return v @ np.diag(w) @ v.T

def inertia(centroids, data):
    n_comp = centroids.shape[0]
    return np.array([[np.linalg.norm(data[i] - centroids[j]) ** 2 for i in range(data.shape[0])] for j in range(n_comp)]).min(axis=0).sum()



if __name__ == '__main__':
    data = load_digits_dataset()
