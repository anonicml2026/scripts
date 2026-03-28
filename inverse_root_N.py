#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from typing import NamedTuple, Optional
from dataclasses import dataclass
import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib.pyplot as plt


# In[2]:


def kernel_K(x, y, a):
    a = np.asarray(a, dtype=float)
    a0, a_rest = a[0], a[1:]

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = np.subtract.outer(np.ravel(x), np.ravel(y))

    i = np.arange(1, a.size)[:, None, None]
    cos_terms = np.cos(i * d)
    K = a0 + np.tensordot(a_rest, cos_terms, axes=(0, 0))

    K = K.reshape(x.size, y.size)
    return K.reshape(np.broadcast(x[..., None], y[None, ...]).shape)


# In[3]:


# build a
def build_coeffs(N, s):
    i = np.arange(0, N+1)
    return (1 + i**2) ** (-s)


# In[4]:


def fourier_design(x, k: int):
    x = np.asarray(x, dtype=float).ravel()              # (n,)
    n = x.size
    Phi = np.empty((n, 2*k + 1), dtype=float)

    # cos(0*x)=1, cos(ix) for i=1..k
    Phi[:, 0] = 1.0
    i = np.arange(1, k+1)                           # (k,)
    Phi[:, 1:k+1] = np.cos(np.outer(x, i))          # cos block
    Phi[:, k+1:]   = np.sin(np.outer(x, i))         # sin block

    return Phi


# In[5]:


def target_fk(x, k, a):
    a = np.asarray(a, float)
    x = np.asarray(x, float)
    ak_sqrt = np.sqrt(a[k])
    return ak_sqrt * (np.cos(k * x) + np.sin(k * x))

def target_sumcos(x, k, a):
    return np.cos(x) + 2*np.cos(2*x) + 3*np.cos(3*x) + 4*np.cos(4*x) + 5*np.cos(5*x)
#    return 1 + np.cos(x) + np.cos(2*x) + np.cos(3*x) + np.cos(4*x) + np.cos(5*x)

def target_sumsin(x, k, a):
    return 1 + np.sin(x) + np.sin(2*x) + np.sin(3*x) + np.sin(4*x) + np.sin(5*x)

def target_cos5(x, k, a):
    return np.cos(5*x)


# In[6]:


def sample_X_eps(N: int, sigma: float, rng: Optional[np.random.Generator] = None):
    rng = np.random.default_rng(rng)
    X   = rng.uniform(0.0, 2*np.pi, size=N)     # Uniform[0, 2π]
    eps = rng.normal(0.0, sigma, size=N)        # N(0, σ^2)
    return X, eps

class TwoDatasets(NamedTuple):
    X: np.ndarray          # shape (N,)
    eps: np.ndarray        # shape (N,)
    y_signal: np.ndarray   # shape (N,)  with f_k(X) + eps
    y_noise: np.ndarray    # shape (N,)  with eps only

def make_datasets(N: int, sigma: float, k: int, a, rng=None, target=target_fk) -> TwoDatasets:    
    X, eps = sample_X_eps(N, sigma, rng)
    y_signal = target(X, k, a) + eps
    y_noise  = eps.copy()                  # same ε, no signal
    return TwoDatasets(X=X, eps=eps, y_signal=y_signal, y_noise=y_noise)


# In[7]:


# ----- residual kernel for empirical P_N -----
def residual_kernel_empirical(x, y, X, a, k: int):
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    X = np.asarray(X, float).ravel()
    N  = X.size
    m  = 2*k + 1

    # basis evaluations
    Phi_X = fourier_design(X, k)        # (N, m)
    Phi_x = fourier_design(x, k)        # (nx, m)
    Phi_y = fourier_design(y, k)        # (ny, m)

    # Gram G and its pseudo-inverse
    G = (Phi_X.T @ Phi_X) / N           # (m, m)
    A = np.linalg.pinv(G)               # (m, m)

    # projection kernel pieces Π(u, X) = Φ(u) A Φ(X)^T
    Pi_xX = Phi_x @ A @ Phi_X.T         # (nx, N)
    Pi_yX = Phi_y @ A @ Phi_X.T         # (ny, N)

    # base kernel blocks
    K_xy = kernel_K(x, y, a)            # (nx, ny)
    K_xX = kernel_K(x, X, a)            # (nx, N)
    K_Xy = kernel_K(X, y, a)            # (N, ny)
    K_XX = kernel_K(X, X, a)            # (N, N)

    # residualization: (I-Π) on both arguments
    T1 = (Pi_xX @ K_Xy) / N             # ∫ Π(x,u) K(u,y) dP_N(u)
    T2 = (K_xX @ Pi_yX.T) / N           # ∫ K(x,v) Π(y,v) dP_N(v)
    T3 = (Pi_xX @ K_XX @ Pi_yX.T) / (N**2)  # double integral

    return K_xy - T1 - T2 + T3


# In[8]:


# ---------- Conditional KRR (linear part in F + KRR with residual kernel) ----------
@dataclass
class ConditionalKRR:
    a: np.ndarray       # kernel coefficients [a0, a1, ...]
    k: int              # Fourier order for F
    lam_krr: float = 2e-4     # λ in KRR

    def fit(self, X, y):
        X = np.asarray(X, float).ravel()
        y = np.asarray(y, float).ravel()
        N = X.size

        # linear regression in F
        Phi = fourier_design(X, self.k)          # (N, 2k+1)
        A = (Phi.T @ Phi) / N
        b = (Phi.T @ y) / N
        self.theta = np.linalg.solve(A, b)       # coefficients in F
        r = y - Phi @ self.theta                 # residuals

        # KRR on residuals with residual kernel
        K_res = residual_kernel_empirical(X, X, X, self.a, self.k)
        self.alpha = np.linalg.solve(K_res + N*self.lam_krr*np.eye(N), r)

        # store training data for prediction
        self.X_train = X
        return self

    def predict(self, x):
        x = np.asarray(x, float).ravel()
        Phi_x = fourier_design(x, self.k)
        Kx = residual_kernel_empirical(x, self.X_train, self.X_train,
                                       self.a, self.k)
        return Phi_x @ self.theta + Kx @ self.alpha


# In[9]:


# --- tail (residual) kernel: drop the first k+1 Fourier coeffs ---
def tail_kernel_KP(x, y, a, k: int):
    a = np.asarray(a, float)
    tail = a[k+1:]                         # keep a_{k+1}, a_{k+2}, ...
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    if tail.size == 0:
        return np.zeros((x.size, y.size), float)
    d = x[:, None] - y[None, :]
    i = np.arange(k+1, k+1+tail.size)[:, None, None]
    return np.tensordot(tail, np.cos(i * d), axes=(0, 0))     # (nx, ny)

# --- KRR with residual kernel on pure-noise data ---
class NoiseKRR:
    def __init__(self, a, k, lam=2e-4):
        self.a = np.asarray(a, float)  # [a0, a1, ..., aM]
        self.k = int(k)                # F has dim 2k+1
        self.lam = float(lam)          # ridge λ

    def fit(self, X, eps):
        X = np.asarray(X, float).ravel()
        eps = np.asarray(eps, float).ravel()
        N = X.size

        K = tail_kernel_KP(X, X, self.a, self.k)
        self.alpha = np.linalg.solve(K + N*self.lam*np.eye(N), eps)
        self.X_train = X
        return self

    def predict(self, x):
        x = np.asarray(x, float).ravel()
        Kx = tail_kernel_KP(x, self.X_train, self.a, self.k)
        return Kx @ self.alpha




# In[24]:


rng = np.random.default_rng(0)
rng_test = np.random.default_rng(42)
k = 5
sigma = 1.0
lam = 0.01
s = 0.0

N_grid = [20, 30, 50, 75, 100, 150, 200, 300, 500]
n_iter = 50

mse = np.empty((len(N_grid), n_iter))
c_con = np.empty((len(N_grid), n_iter))

print("Done: ", end="")
for n, N in enumerate(N_grid):
    for i in range(n_iter):
        a = build_coeffs(300, s=s)
        data = make_datasets(N, sigma, k+1, a, rng)

        model = ConditionalKRR(a=a, k=k, lam_krr=lam).fit(data.X, data.y_signal)

        x_grid = rng.uniform(0.0, 2*np.pi, size=300)
        y_true = target_fk(x_grid, k+1, a)
        y_pred = model.predict(x_grid)

        model_noise = NoiseKRR(a=a, k=k, lam=lam).fit(data.X, data.y_signal)
        fhat_noise = model_noise.predict(x_grid)

        mse[n, i] = np.mean((y_pred - y_true)**2)
        c_con[n, i] = np.mean((y_pred - fhat_noise)**2)
    print(" ... "+str(round(float(n+1) / len(N_grid)*100)) + "%", end=" ")


# #### $c_\text{con}$ vs $N$ plot




x = np.log(N_grid)
y = np.log(np.quantile(c_con, 0.8, axis=1))

slope, intercept = np.polyfit(x[2:], y[2:], 1)

plt.figure(figsize=(7,4))
plt.plot(x, y, marker='o', linestyle='None', label=r'$\widehat{c}_{\text{con}}$')
plt.plot(x, slope*x + intercept, color="red", 
         label=f"slope$={slope:.2f}$", linestyle="--")
plt.xlabel(r'$\log N$')
plt.ylabel(r'$\log \widehat{c_{\text{con}}}$')
plt.legend()
plt.title(r'$k=$'+str(k)+r', $\sigma=$'+str(sigma)+r', $\lambda=$'+str(lam)+r', $K(x,y)=\sum_{i=1}^{300}\cos(i(x-y))$, target$=\cos((k+1)x)+\sin((k+1)x)$')
plt.savefig('c_con(N)_worst_'+'target=f_{k+1}.png', dpi=600)





