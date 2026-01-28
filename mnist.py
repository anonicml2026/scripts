#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from numpy.linalg import solve

# -------------------
# 1. Load MNIST 7 vs 9
# -------------------
# transform: tensor + crop + normalize to [0,1]
transform = transforms.Compose([
    transforms.ToTensor()
])

mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
X, y = [], []



for img, label in mnist_train:
    if label in [7, 9]:
        arr = img.numpy().squeeze()  # (28,28)
        arr = arr[2:-2, 2:-2]        # crop to (24,24)
        arr = arr.astype(np.float32)
        X.append(arr.flatten())
        y.append(+1 if label == 7 else -1)

X = np.array(X)
y = np.array(y)

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

pixel_mean = X_train_full.mean(axis=0)  # shape (d,)
X_train = X_train_full - pixel_mean
X_test  = X_test_full - pixel_mean

N_train, d = X_train.shape
N_test = X_test.shape[0]

print(f"Train size: {N_train}, Test size: {N_test}, Dim: {d}")

# -------------------
# 2. Random Feature Ridge Regression
# -------------------
def build_random_features(X, omega, b):
    x = X @ omega.T + b[None, :]
    return np.cos(X @ omega.T + b[None, :]) #np.tanh(x)#x * (x > 0) #

def run_rfrr(X_train, y_train, X_test, y_test, k, m=10000, sigma=1.0, lam=0.1):
    """
    Run RFRR with k unpenalized and m penalized features.
    """
    N, d = X_train.shape

    # pre-sample omegas and phases
    omega = np.random.normal(scale=1/sigma, size=(k + m, d))
#    b = np.random.uniform(-1, 1, size=(k + m,))
    b = np.random.uniform(0, 2*np.pi, size=(k + m,))

    # indices
    idx_phi = np.arange(k)
    idx_psi = np.arange(k, k + m)

    # build features
    Phi_train = build_random_features(X_train, omega[idx_phi], b[idx_phi]) if k > 0 else np.zeros((N,0))
    Psi_train = build_random_features(X_train, omega[idx_psi], b[idx_psi]) / np.sqrt(m)
    A_train = np.hstack([Phi_train, Psi_train])

    Phi_test = build_random_features(X_test, omega[idx_phi], b[idx_phi]) if k > 0 else np.zeros((X_test.shape[0],0))
    Psi_test = build_random_features(X_test, omega[idx_psi], b[idx_psi]) / np.sqrt(m)
    A_test = np.hstack([Phi_test, Psi_test])

    # Ridge regression with selective penalty
    penalty = np.concatenate([np.zeros(k), np.full(m, lam)])
    H = (A_train.T @ A_train) / N
    b_vec = (A_train.T @ y_train) / N

    W = solve(H + np.diag(penalty), b_vec)

    # predictions
    y_pred_train = A_train @ W
    y_pred_test  = A_test @ W

    mse_train = np.mean((y_pred_train - y_train)**2)
    mse_test  = np.mean((y_pred_test  - y_test)**2)
    return mse_train, mse_test

# -------------------
# 3. Experiment
# -------------------
n_list = np.linspace(0, 2000, 20, dtype=int)
m = 500
sigma = 1.0
lam = 0.1

train_errors = []
test_errors = []

for k in n_list:
    mse_train, mse_test = run_rfrr(X_train, y_train, X_test, y_test,
                                   k=k, m=m, sigma=sigma, lam=lam)
    train_errors.append(mse_train)
    test_errors.append(mse_test)
    print(f"k={k:3d} | Train MSE={mse_train:.4f}, Test MSE={mse_test:.4f}")

# -------------------
# 4. Plot results
# -------------------
plt.figure(figsize=(6,4))
plt.plot(n_list, test_errors, marker="o", label="Test MSE")
plt.plot(n_list, train_errors, marker="s", label="Train MSE", alpha=0.6)
plt.xlabel("Number of unpenalized features $k$")
plt.ylabel("MSE")
plt.title("RFRR on MNIST 7 vs 9 ($\lambda=0.1$), cos")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('mnist_relu.png', dpi=600, bbox_inches="tight")
plt.show()


# In[ ]:




