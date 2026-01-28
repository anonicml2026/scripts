#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib.pyplot as plt
from scipy.linalg import solve
from tqdm import trange


# In[2]:


# 1. Experiment parameters
d = 10                # ambient dimension
N_train = 500         # number of training points
N_test  = 500         # number of test points
sigma = 1.0           # Gaussian kernel bandwidth parameter
eps = 1e-2            # target training error for memorization threshold
overfit_tol = 10*eps  # tolerance for catastrophic overfit threshold

# choice of target function f: here a simple smooth function on the sphere
def target_function(x):
    # e.g. depends on first two coordinates
    return np.sin(x[:,0]) + 0.5 * np.cos(2 * x[:,1])

# grid of λ values to scan (log‑spaced)
lambdas = np.logspace(-6, -3, 50)

# values of n (number of unpenalized random Fourier features)
n_list = np.linspace(0, 300, 10, dtype=int) #np.array([  0,  10,  20,  50, 100, 200, 400 ])

# choose m large (residual features)
m = 2000


# In[3]:


# 2. generate train and test data on the sphere
def sample_on_sphere(N, dim):
    X = np.random.normal(size=(N, dim))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

X_train = sample_on_sphere(N_train, d)
y_train = target_function(X_train) + 0.2*np.random.normal(size=N_train)
X_test  = sample_on_sphere(N_test,  d)
y_test  = target_function(X_test) + 0.2*np.random.normal(size=N_train)


# In[4]:


# 3. helper to build random features
def build_random_features(X, omega, b):
    # X: (N, d), omega: (p, d), b: (p,)
    return np.cos(X.dot(omega.T) + b[None, :])

# 4. pre‑sample all omegas and bs for both φ and ψ
omega = np.random.normal(scale=1/sigma, size=(n_list.max() + m, d))
b     = np.random.uniform(0, 2*np.pi, size=(n_list.max() + m,))

# 5. containers for the two curves
lambda_mem = np.zeros_like(n_list, dtype=float)
lambda_cat = np.zeros_like(n_list, dtype=float)


# In[5]:


for idx, n in enumerate(n_list):
    # indices for the two blocks of features
    idx_phi = np.arange(n)
    idx_psi = np.arange(n, n + m)

    # build feature matrices
    Phi_train = build_random_features(X_train, omega[idx_phi], b[idx_phi])
    Psi_train = build_random_features(X_train, omega[idx_psi], b[idx_psi]) / np.sqrt(m)
    A_train = np.hstack([Phi_train, Psi_train])

    Phi_test = build_random_features(X_test, omega[idx_phi], b[idx_phi])
    Psi_test = build_random_features(X_test, omega[idx_psi], b[idx_psi]) / np.sqrt(m)
    A_test  = np.hstack([Phi_test,  Psi_test])

    N = float(N_train)
    H = (A_train.T @ A_train) / N
    b_vec = (A_train.T @ y_train) / N

    train_errors = []
    test_errors  = []

    # scan lambdas
    for lam in lambdas:
        # build penalty: zero for first n coords, lam for last m coords
        penalty = np.concatenate([np.zeros(n), np.full(m, lam)])
        # solve (H + diag(penalty)) x = b_vec
        W = solve(H + np.diag(penalty), b_vec, assume_a='pos')
        #W = np.linalg.pinv(H + np.diag(penalty), hermitian=True) @ b_vec
        # predictions
        y_pred_train = A_train @ W
        y_pred_test  = A_test  @ W
        train_mse = np.mean((y_pred_train - y_train)**2)
        test_mse  = np.mean((y_pred_test  - y_test )**2)

        train_errors.append(train_mse)
        test_errors.append(test_mse)

    train_errors = np.array(train_errors)
    test_errors  = np.array(test_errors)

    # λ_mem(n): largest λ s.t. train_mse < eps
    mask_mem = train_errors < eps
    lambda_mem[idx] = lambdas[mask_mem].max() if mask_mem.any() else np.nan

    # λ_cat(n): smallest λ s.t. test_mse - train_mse < overfit_tol
    diff = test_errors - train_errors
    mask_cat = diff < overfit_tol
    lambda_cat[idx] = lambdas[mask_cat].min() if mask_cat.any() else np.nan


# In[6]:


# 6. plot the results
plt.figure(figsize=(6,4))
plt.plot(n_list, lambda_mem, marker='o', label=r'$\lambda_{\mathrm{mem}}(k)$')
plt.plot(n_list, lambda_cat, marker='s', label=r'$\lambda_{\mathrm{cat}}(k)$')
#plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$k$ (number of unpenalized features)')
plt.ylabel(r'$\lambda$ threshold')
plt.legend()
plt.title('Memorization vs Catastrophic Overfit thresholds')
plt.grid(True, which='both', ls='--', alpha=0.7)
plt.tight_layout()
plt.savefig('CPD Mem-Cat.png', dpi=600)

