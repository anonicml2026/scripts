#!/usr/bin/env python
# coding: utf-8

# In[22]:


import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve


# In[23]:


# Experiment parameters
d = 5
N_train = 500
N_test = 500
sigma = 1.0
lam = 1e-2           # fixed lambda
m = 2000             # number of penalized features


# In[24]:


# Target function
def target_function(x):
    return np.sin(x[:,0]) + 0.5 * np.cos(2 * x[:,1])

# Sample on sphere
def sample_on_sphere(N, dim):
    X = np.random.normal(size=(N, dim))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

# Generate data
X_train = sample_on_sphere(N_train, d)
y_train = target_function(X_train) + 0.2*np.random.normal(size=N_train)
X_test = sample_on_sphere(N_test, d)
y_test = target_function(X_test) + 0.2*np.random.normal(size=N_test)


# ### Gaussian Kernel

# In[28]:


# helper to build random features
def build_random_features(X, omega, b):
    # X: (N, d), omega: (p, d), b: (p,)
    return np.cos(X.dot(omega.T) + b[None, :])

# Random feature parameters
max_n = 200
omega = np.random.normal(scale=1/sigma, size=(max_n + m, d))
b = np.random.uniform(0, 2 * np.pi, size=(max_n + m,))

# Define n_list
n_list = np.array([4*i for i in range(0,51)]) #np.array([0, 10, 20, 50, 100, 200, 400, 600])

train_mse_list = []
test_mse_list = []


# In[29]:


for n in n_list:
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

    # Solve with pseudo-inverse
    penalty = np.concatenate([np.zeros(n), np.full(m, lam)])
    W = solve(H + np.diag(penalty), b_vec, assume_a='pos')
    # solve(H + np.diag(penalty), b_vec, assume_a='pos')
    # np.linalg.pinv(H + np.diag(penalty), hermitian=True) @ b_vec
    # np.linalg.pinv(H + np.diag(penalty)) @ b_vec

    # Compute MSEs
    y_pred_train = A_train @ W
    y_pred_test = A_test @ W
    train_mse = np.mean((y_pred_train - y_train) ** 2)
    test_mse = np.mean((y_pred_test - y_test) ** 2)

    train_mse_list.append(train_mse)
    test_mse_list.append(test_mse)


# In[30]:


# Plot with dual y-axes, different colors and legend
fig, ax1 = plt.subplots(figsize=(6, 4))
ax2 = ax1.twinx()

ln1 = ax1.plot(n_list, 
               train_mse_list, 
               marker='o', 
               markersize=2,
               color='tab:blue', 
               label='Train MSE')
ln2 = ax2.plot(n_list, 
               test_mse_list, 
               marker='s', 
               markersize=2,
               color='tab:orange', 
               label='Test MSE')

ax1.set_xlabel('$k$ (number of unpenalized features)')
ax1.set_ylabel('Train MSE', color='tab:blue')
ax2.set_ylabel('Test MSE', color='tab:orange')

# Combine legends
lines = ln1 + ln2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.title(r'Train vs Test MSE vs $k$ at $d=5, \lambda = 10^{-2}, \cos$')
ax1.grid(True)
fig.tight_layout()
plt.savefig('train_test_mse.png', dpi=600)


# ### NNGP Kernel

# In[25]:


def relu(x_array):
    return np.maximum(0, x_array)

# helper to build random features
def build_random_features(X, omega, b):
    # X: (N, d), omega: (p, d), b: (p,)
    return relu(X.dot(omega.T) + b[None, :])

# Random feature parameters
max_n = 400
omega = np.random.normal(scale=1, size=(max_n + m, d))
b = np.random.uniform(-1, 1, size=(max_n + m,))

# Define n_list
n_list = np.array([8*i for i in range(0,51)]) #np.array([0, 10, 20, 50, 100, 200, 400, 600])

train_mse_list = []
test_mse_list = []


# In[26]:


for n in n_list:
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

    # Solve with pseudo-inverse
    penalty = np.concatenate([np.zeros(n), np.full(m, lam)])
    W = np.linalg.pinv(H + np.diag(penalty)) @ b_vec
    # solve(H + np.diag(penalty), b_vec, assume_a='pos')
    # np.linalg.pinv(H + np.diag(penalty), hermitian=True) @ b_vec
    # np.linalg.pinv(H + np.diag(penalty)) @ b_vec

    # Compute MSEs
    y_pred_train = A_train @ W
    y_pred_test = A_test @ W
    train_mse = np.mean((y_pred_train - y_train) ** 2)
    test_mse = np.mean((y_pred_test - y_test) ** 2)

    train_mse_list.append(train_mse)
    test_mse_list.append(test_mse)


# In[27]:


# Plot with dual y-axes, different colors and legend
fig, ax1 = plt.subplots(figsize=(6, 4))
ax2 = ax1.twinx()

ln1 = ax1.plot(n_list, 
               train_mse_list, 
               marker='o', 
               markersize=2,
               color='tab:blue', 
               label='Train MSE')
ln2 = ax2.plot(n_list, 
               test_mse_list, 
               marker='s', 
               markersize=2,
               color='tab:orange', 
               label='Test MSE')

ax1.set_xlabel('$k$ (number of unpenalized features)')
ax1.set_ylabel('Train MSE', color='tab:blue')
ax2.set_ylabel('Test MSE', color='tab:orange')

# Combine legends
lines = ln1 + ln2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.title(r'Train vs Test MSE vs $k$ at $d=5, \lambda = 10^{-2}$, ReLU')
ax1.grid(True)
fig.tight_layout()
plt.savefig('train_test_mse_nngp.png', dpi=600)


# In[10]:


def tanh(x_array):
    return np.tanh(x_array)

# helper to build random features
def build_random_features(X, omega, b):
    # X: (N, d), omega: (p, d), b: (p,)
    return np.tanh(X.dot(omega.T) + b[None, :])

# Random feature parameters
max_n = 400
omega = np.random.normal(scale=1, size=(max_n + m, d))
b = np.random.uniform(-1, 1, size=(max_n + m,))

# Define n_list
n_list = np.array([8*i for i in range(0,51)]) #np.array([0, 10, 20, 50, 100, 200, 400, 600])

train_mse_list = []
test_mse_list = []


# In[11]:


for n in n_list:
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

    # Solve with pseudo-inverse
    penalty = np.concatenate([np.zeros(n), np.full(m, lam)])
    W = np.linalg.pinv(H + np.diag(penalty)) @ b_vec
    # solve(H + np.diag(penalty), b_vec, assume_a='pos')
    # np.linalg.pinv(H + np.diag(penalty), hermitian=True) @ b_vec
    # np.linalg.pinv(H + np.diag(penalty)) @ b_vec

    # Compute MSEs
    y_pred_train = A_train @ W
    y_pred_test = A_test @ W
    train_mse = np.mean((y_pred_train - y_train) ** 2)
    test_mse = np.mean((y_pred_test - y_test) ** 2)

    train_mse_list.append(train_mse)
    test_mse_list.append(test_mse)


# In[12]:


# Plot with dual y-axes, different colors and legend
fig, ax1 = plt.subplots(figsize=(6, 4))
ax2 = ax1.twinx()

ln1 = ax1.plot(n_list, 
               train_mse_list, 
               marker='o', 
               markersize=2,
               color='tab:blue', 
               label='Train MSE')
ln2 = ax2.plot(n_list, 
               test_mse_list, 
               marker='s', 
               markersize=2,
               color='tab:orange', 
               label='Test MSE')

ax1.set_xlabel('$k$ (number of unpenalized features)')
ax1.set_ylabel('Train MSE', color='tab:blue')
ax2.set_ylabel('Test MSE', color='tab:orange')

# Combine legends
lines = ln1 + ln2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.title(r'Train vs Test MSE vs $k$ at $d=5, \lambda = 10^{-2}, \tanh$')
ax1.grid(True)
fig.tight_layout()
plt.savefig('train_test_mse_nngp.png', dpi=600)


# In[ ]:




