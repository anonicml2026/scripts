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
    return 1 + np.cos(x) + np.cos(2*x) + np.cos(3*x) + np.cos(4*x) + np.cos(5*x)

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


# ## Dependence on N

# ### Target = $f_k$

# In[67]:


rng = np.random.default_rng(0)
rng_test = np.random.default_rng(42)
k = 5
sigma = 1.0
lam = .1
s = .75

N_grid = [20, 30, 50, 75, 100, 150, 200, 300, 500, 1000]
n_iter = 10

mse = np.empty((len(N_grid), n_iter))
c_con = np.empty((len(N_grid), n_iter))

print("Done: ", end="")
for n, N in enumerate(N_grid):
    for i in range(n_iter):
        a = build_coeffs(20, s=s)
        data = make_datasets(N, sigma, k, a, rng)

        model = ConditionalKRR(a=a, k=k, lam_krr=lam).fit(data.X, data.y_signal)

        x_grid = rng.uniform(0.0, 2*np.pi, size=300)
        y_true = target_fk(x_grid, k, a)
        y_pred = model.predict(x_grid)

        model_noise = NoiseKRR(a=a, k=k, lam=lam).fit(data.X, data.y_noise)
        fhat_noise = model_noise.predict(x_grid)   # this is \hat f_noise(x)
        true_plus_noise = y_true + fhat_noise

        mse[n, i] = np.mean((y_pred - y_true)**2)
        c_con[n, i] = np.mean((y_pred - true_plus_noise)**2)
    print(" ... "+str(round(float(n+1) / len(N_grid)*100)) + "%", end=" ")


# #### $c_\text{con}$ vs $N$ plot

# In[68]:


x = np.log(N_grid)
y = np.log(c_con.mean(axis=1))

slope, intercept = np.polyfit(x[2:], y[2:], 1)

plt.figure(figsize=(6,4))
plt.plot(x, y, marker='o', linestyle='None', label=r'$\widehat{c}_{\text{con}}$')
plt.plot(x, slope*x + intercept, color="red", 
         label=f"slope$={slope:.2f}$", linestyle="--")
plt.xlabel(r'$\log N$')
plt.ylabel(r'$\log \widehat{c_{\text{con}}}$')
plt.legend()
plt.title(r'$k=$'+str(k)+r', $\sigma=$'+str(sigma)+r', $\lambda=$'+str(lam)+', $s=$'+str(s)+r', target$=\frac{\cos(kx)+\sin(kx)}{k^{0.75}}$')
plt.savefig('c_con(N)_target=fk.png', dpi=600)


# In[12]:


x = np.log(N_grid)
y = np.log(mse.mean(axis=1))

slope, intercept = np.polyfit(x[2:], y[2:], 1)

plt.figure(figsize=(6,4))
plt.plot(x, y, marker='o', linestyle='None', label=r'MSE')
plt.plot(x, slope*x + intercept, color="red", 
         label=f"slope$={slope:.2f}$", linestyle="--")
plt.xlabel(r'$\log N$')
plt.ylabel(r'$\log \text{MSE}$')
plt.legend()
plt.title(r'$k=$'+str(k)+r', $\sigma=$'+str(sigma)+r', $\lambda=$'+str(lam)+', $s=$'+str(s)+r', target$=f_{k}$')
plt.savefig('mse(N)_target=fk.png', dpi=600)


# #### $\hat{f}$ vs $f_k$

# In[13]:


N = 100
a = build_coeffs(N, s=s)
data = make_datasets(N, sigma, k, a, rng)
model = ConditionalKRR(a=a, k=k, lam_krr=lam).fit(data.X, data.y_signal)
model_noise = NoiseKRR(a=a, k=k, lam=lam).fit(data.X, data.y_signal)

x_grid = np.linspace(0, 2*np.pi, 300, endpoint=False)
y_true = target_fk(x_grid, k, a)
y_pred = model.predict(x_grid)
fhat_noise = model_noise.predict(x_grid)
true_plus_noise = y_true + fhat_noise

plt.figure(figsize=(6,4))
plt.plot(x_grid, y_true, label=r'$f_{'+str(k)+r'}(x)$', color='black')
plt.plot(x_grid, y_pred, label=r'$\hat{f}(x)$', linestyle='--', linewidth=.5, color='red')
plt.plot(x_grid, true_plus_noise, label=r'$f_{'+str(k)+r'}(x)+\hat{f}_{\text{noise}}(x)$', linestyle='--', linewidth=.5, color='blue')
plt.xlabel(r'$x$')
plt.legend()
plt.title(r'$k=$'+str(k)+r', $\sigma=$'+str(sigma)+r', $\lambda=$'+str(lam)+', $s=$'+str(s)+r', target$=f_{k}$, '+r"$N=$"+str(N))
plt.savefig('true_vs_pred_N='+str(N)+'_target=fk.png', dpi=600)


# ### Target = $f_{k+1}$

# In[69]:


rng = np.random.default_rng(0)
rng_test = np.random.default_rng(42)
k = 5
sigma = 1.0
lam = 0.1
s = 0.75

N_grid = [20, 30, 50, 75, 100, 150, 200, 300, 500]
n_iter = 20

mse = np.empty((len(N_grid), n_iter))
c_con = np.empty((len(N_grid), n_iter))

print("Done: ", end="")
for n, N in enumerate(N_grid):
    for i in range(n_iter):
        a = build_coeffs(20, s=s)
        data = make_datasets(N, sigma, k+1, a, rng)

        model = ConditionalKRR(a=a, k=k, lam_krr=lam).fit(data.X, data.y_signal)

        x_grid = rng_test.uniform(0.0, 2*np.pi, size=300)
        y_true = target_fk(x_grid, k+1, a)
        y_pred = model.predict(x_grid)

        model_noise = NoiseKRR(a=a, k=k, lam=lam).fit(data.X, data.y_signal)
        fhat_noise = model_noise.predict(x_grid)

        mse[n, i] = np.mean((y_pred - y_true)**2)
        c_con[n, i] = np.mean((y_pred - fhat_noise)**2)
    print(" ... "+str(round(float(n+1) / len(N_grid)*100)) + "%", end=" ")


# In[66]:


x = np.log(N_grid)
y = np.log(c_con.mean(axis=1))

slope, intercept = np.polyfit(x[2:], y[2:], 1)

plt.figure(figsize=(6,4))
plt.plot(x, y, marker='o', linestyle='None', label=r'$\widehat{c}_{\text{con}}$')
plt.plot(x, slope*x + intercept, color="red", 
         label=f"slope$={slope:.2f}$", linestyle="--")
plt.xlabel(r'$\log N$')
plt.ylabel(r'$\log \widehat{c_{\text{con}}}$')
plt.legend()
plt.title(r'$k=$'+str(k)+r', $\sigma=$'+str(sigma)+r', $\lambda=$'+str(lam)+', $s=$'+str(s)+r', target$=\frac{\cos((k+1)x)+\sin((k+1)x)}{(k+1)^{0.75}}$')
plt.savefig('c_con(N)_'+'target=f(k+1).png', dpi=600)


# ### Target = $f_{2k}$

# In[70]:


rng = np.random.default_rng(0)
rng_test = np.random.default_rng(42)
k = 5
sigma = 1.0
lam = .1
s = 0.75

N_grid = [20, 30, 50, 75, 100, 150, 200, 300, 500]
n_iter = 50

mse = np.empty((len(N_grid), n_iter))
c_con = np.empty((len(N_grid), n_iter))

print("Done: ", end="")
for n, N in enumerate(N_grid):
    for i in range(n_iter):
        a = build_coeffs(20, s=s)
        data = make_datasets(N, sigma, 2*k, a, rng)

        model = ConditionalKRR(a=a, k=k, lam_krr=lam).fit(data.X, data.y_signal)

        x_grid = rng.uniform(0.0, 2*np.pi, size=300)
        y_true = target_fk(x_grid, 2*k, a)
        y_pred = model.predict(x_grid)

        model_noise = NoiseKRR(a=a, k=k, lam=lam).fit(data.X, data.y_signal)
        fhat_noise = model_noise.predict(x_grid)

        mse[n, i] = np.mean((y_pred - y_true)**2)
        c_con[n, i] = np.mean((y_pred - fhat_noise)**2)
    print(" ... "+str(round(float(n+1) / len(N_grid)*100)) + "%", end=" ")


# #### $c_\text{con}$ vs $N$ plot

# In[71]:


x = np.log(N_grid)
y = np.log(c_con.mean(axis=1))

slope, intercept = np.polyfit(x[2:], y[2:], 1)

plt.figure(figsize=(6,4))
plt.plot(x, y, marker='o', linestyle='None', label=r'$\widehat{c}_{\text{con}}$')
plt.plot(x, slope*x + intercept, color="red", 
         label=f"slope$={slope:.2f}$", linestyle="--")
plt.xlabel(r'$\log N$')
plt.ylabel(r'$\log \widehat{c_{\text{con}}}$')
plt.legend()
plt.title(r'$k=$'+str(k)+r', $\sigma=$'+str(sigma)+r', $\lambda=$'+str(lam)+', $s=$'+str(s)+r', target$=\frac{\cos(2kx)+\sin(2kx)}{(2k)^{0.75}}$')
plt.savefig('c_con(N)_'+'target=f2k.png', dpi=600)


# #### $\hat{f}$ vs $f_{2k}$

# In[18]:


N = 300
a = build_coeffs(N, s=s)
data = make_datasets(N, sigma, 2*k, a, rng)
model = ConditionalKRR(a=a, k=k, lam_krr=lam).fit(data.X, data.y_signal)
model_noise = NoiseKRR(a=a, k=k, lam=lam).fit(data.X, data.y_signal)

x_grid = np.linspace(0, 2*np.pi, 300, endpoint=False)
y_true = target_fk(x_grid, 2*k, a)
y_pred = model.predict(x_grid)
fhat_noise = model_noise.predict(x_grid)

plt.figure(figsize=(6,4))
plt.plot(x_grid, y_true, label=r'$f_{'+str(2*k)+r'}(x)$', color='black')
plt.plot(x_grid, y_pred, label=r'$\hat{f}(x)$', linestyle='--', linewidth=.5, color='red')
plt.plot(x_grid, fhat_noise, label=r'$\hat{f}_{\text{noise}}(x)$', linestyle='--', linewidth=.5, color='blue')
plt.plot(data.X, data.y_
plt.xlabel(r'$x$')
plt.legend()
plt.title(r'$k=$'+str(k)+r', $\sigma=$'+str(sigma)+r', $\lambda=$'+str(lam)+', $s=$'+str(s)+r', target$=f_{2k}$'+r", $N=$"+str(N))
plt.savefig('true_vs_pred_N='+str(N)+'_target=f2k.png', dpi=600)


# #### target = $\cos(5x)$, but $k=2$

# In[ ]:


rng = np.random.default_rng(0)
rng_test = np.random.default_rng(42)
k = 2
sigma = 1.0
lam = .1
s = .75

N_grid = [20, 30, 50, 75, 100, 150, 200, 300]
n_iter = 10

mse = np.empty((len(N_grid), n_iter))
c_con = np.empty((len(N_grid), n_iter))

print("Done: ", end="")
for n, N in enumerate(N_grid):
    for i in range(n_iter):
        a = build_coeffs(N, s=s)
        data = make_datasets(N, sigma, k, a, rng, target=target_cos5)

        model = ConditionalKRR(a=a, k=k, lam_krr=lam).fit(data.X, data.y_signal)

        x_grid = rng.uniform(0.0, 2*np.pi, size=300)
        y_true = target_cos5(x_grid, k, a)
        y_pred = model.predict(x_grid)

        model_noise = NoiseKRR(a=a, k=k, lam=lam).fit(data.X, data.y_signal)
        fhat_noise = model_noise.predict(x_grid)

        mse[n, i] = np.mean((y_pred - y_true)**2)
        c_con[n, i] = np.mean((y_pred - fhat_noise)**2)
    print(" ... "+str(round(float(n+1) / len(N_grid)*100)) + "%", end=" ")


# In[ ]:


x = np.log(N_grid)
y = np.log(c_con.mean(axis=1))

slope, intercept = np.polyfit(x, y, 1)

plt.figure(figsize=(6,4))
plt.plot(x, y, marker='o', linestyle='None', label=r'$\widehat{c}_{\text{con}}$')
plt.plot(x, slope*x + intercept, color="red", 
         label=f"slope$={slope:.2f}$", linestyle="--")
plt.xlabel(r'$\log N$')
plt.ylabel(r'$\log \widehat{c_{\text{con}}}$')
plt.legend()
plt.title(r'$k=$'+str(k)+r', $\sigma=$'+str(sigma)+r', $\lambda=$'+str(lam)+', $s=$'+str(s)+r', target$=\cos(5x)$')
plt.savefig('c_con(N)_'+'target=cos5x.png', dpi=600)


# In[ ]:


N = 100
a = build_coeffs(N, s=s)
data = make_datasets(N, sigma, k, a, rng, target=target_cos5)
model = ConditionalKRR(a=a, k=k, lam_krr=lam).fit(data.X, data.y_signal)
model_noise = NoiseKRR(a=a, k=k, lam=lam).fit(data.X, data.y_signal)

x_grid = np.linspace(0, 2*np.pi, 300, endpoint=False)
y_true = target_cos5(x_grid, k, a)
y_pred = model.predict(x_grid)
fhat_noise = model_noise.predict(x_grid)

plt.figure(figsize=(6,4))
plt.plot(x_grid, y_true, label=r'target', color='black')
plt.plot(x_grid, y_pred, label=r'$\hat{f}(x)$', linestyle='--', linewidth=.5, color='red')
plt.plot(x_grid, fhat_noise, label=r'$\hat{f}_{\text{noise}}(x)$', linestyle='--', linewidth=.5, color='blue')
plt.xlabel(r'$x$')
plt.legend()
plt.title(r'$k=$'+str(k)+r', $\sigma=$'+str(sigma)+r', $\lambda=$'+str(lam)+', $s=$'+str(s)+r', target$=\cos(5x)$'+r", $N=$"+str(N))
plt.savefig('true_vs_pred_N='+str(N)+'_target=cos5x.png', dpi=600)


# ## Dependence on k

# #### target = $f_k$

# In[72]:


rng = np.random.default_rng(0)
rng_test = np.random.default_rng(42)
N = 100
sigma = 1.0
lam = .1
s = .75

k_grid = [i for i in range(0, 16)]
n_iter = 10

mse = np.empty((len(k_grid), n_iter))
c_con = np.empty((len(k_grid), n_iter))

print("Done: ", end="")
for ix, k in enumerate(k_grid):
    for i in range(n_iter):
        a = build_coeffs(20, s=s)
        data = make_datasets(N, sigma, k, a, rng)

        model = ConditionalKRR(a=a, k=k, lam_krr=lam).fit(data.X, data.y_signal)

        x_grid = rng.uniform(0.0, 2*np.pi, size=300)
        y_true = target_fk(x_grid, k, a)
        y_pred = model.predict(x_grid)

        model_noise = NoiseKRR(a=a, k=k, lam=lam).fit(data.X, data.y_noise)
        fhat_noise = model_noise.predict(x_grid)
        true_plus_noise = y_true + fhat_noise

        mse[ix, i] = np.mean((y_pred - y_true)**2)
        c_con[ix, i] = np.mean((y_pred - true_plus_noise)**2)
    print(" ... "+str(round(float(ix+1) / len(k_grid)*100)) + "%", end=" ")


# In[73]:


x = k_grid
y = c_con.mean(axis=1)

slope, intercept = np.polyfit(x, y, 1)

plt.figure(figsize=(6,4))
plt.plot(x, y, marker='o', linestyle='None')
plt.plot(x, slope*np.array(x, dtype='float') + intercept, color="red", 
         label=r"linear fit", linestyle="--")
plt.xlabel(r'$k$')
plt.ylabel(r'$\widehat{c_\text{con}}$')
plt.legend()
plt.title(r'$N=$'+str(N)+r', $\sigma=$'+str(sigma)+r', $\lambda=$'+str(lam)+', $s=$'+str(s)+r', target$=\frac{\cos(kx)+\sin(kx)}{k^{0.75}}$')
plt.savefig('c_con(k)_target=fk.png', dpi=600)


# In[ ]:


N = 100
k = 0
a = build_coeffs(N, s=s)
data = make_datasets(N, sigma, k, a, rng)
model = ConditionalKRR(a=a, k=k, lam_krr=lam).fit(data.X, data.y_signal)

x_grid = np.linspace(0, 2*np.pi, 300, endpoint=False)
y_true = target_fk(x_grid, k, a)
y_pred = model.predict(x_grid)

plt.figure(figsize=(6,4))
plt.plot(x_grid, y_true, label=r'$f_'+str(k)+r'(x)$', color='black')
plt.plot(x_grid, y_pred, label=r'$\hat{f}(x)$', linestyle='--', linewidth=.5, color='red')
plt.plot(data.X, data.y_signal, label=r"$f_"+str(k)+r"(x_i)+\epsilon_i$", marker='o', markersize=2, linestyle='None', color='blue')
plt.xlabel(r'$x$')
plt.legend()
plt.title(r'$N=$'+str(N)+r', $\sigma=$'+str(sigma)+r', $\lambda=$'+str(lam)+', $s=$'+str(s)+r', target$=f_'+str(k)+r'$')
plt.savefig('true_vs_pred_N='+str(N)+'_target=fk.png', dpi=600)


# In[ ]:


N = 100
k = 14
a = build_coeffs(N, s=s)
data = make_datasets(N, sigma, k, a, rng)
model = ConditionalKRR(a=a, k=k, lam_krr=lam).fit(data.X, data.y_signal)

x_grid = np.linspace(0, 2*np.pi, 300, endpoint=False)
y_true = target_fk(x_grid, k, a)
y_pred = model.predict(x_grid)

plt.figure(figsize=(6,4))
plt.plot(x_grid, y_true, label=r'$f_{'+str(k)+r'}(x)$', color='black')
plt.plot(x_grid, y_pred, label=r'$\hat{f}(x)$', linestyle='--', linewidth=.5, color='red')
#plt.plot(x_grid, true_plus_noise, label=r'$f_{2k}(x)+\hat{f}_{\text{noise}}(x)$', linestyle='--', linewidth=.5, color='blue')
plt.plot(data.X, data.y_signal, label=r"$f_{"+str(k)+r"}(x_i)+\epsilon_i$", marker='o', markersize=2, linestyle='None', color='blue')
plt.xlabel(r'$x$')
plt.legend()
plt.title(r'$N=$'+str(N)+r', $\sigma=$'+str(sigma)+r', $\lambda=$'+str(lam)+', $s=$'+str(s)+r', target$=f_{'+str(k)+r'}$')
plt.savefig('true_vs_pred_N='+str(N)+'_target=fk.png', dpi=600)


# #### target = $f_{2k}$

# In[78]:


rng = np.random.default_rng(0)
rng_test = np.random.default_rng(42)
N = 100
sigma = 1.0
lam = 0.1
s = 0.75

k_grid = [i for i in range(0, 16)]
n_iter = 10

mse = np.empty((len(k_grid), n_iter))
c_con = np.empty((len(k_grid), n_iter))

print("Done: ", end="")
for ix, k in enumerate(k_grid):
    for i in range(n_iter):
        a = build_coeffs(40, s=s)
        data = make_datasets(N, sigma, 2*k, a, rng)

        model = ConditionalKRR(a=a, k=k, lam_krr=lam).fit(data.X, data.y_signal)

        x_grid = rng.uniform(0.0, 2*np.pi, size=300)
        y_true = target_fk(x_grid, 2*k, a)
        y_pred = model.predict(x_grid)

        model_noise = NoiseKRR(a=a, k=k, lam=lam).fit(data.X, data.y_noise)
        fhat_noise = model_noise.predict(x_grid)   # this is \hat f_noise(x)
        true_plus_noise = y_true + fhat_noise

        mse[ix, i] = np.mean((y_pred - y_true)**2)
        c_con[ix, i] = np.mean((y_pred - true_plus_noise)**2)
    print(" ... "+str(round(float(ix+1) / len(k_grid)*100)) + "%", end=" ")


# In[80]:


x = k_grid
y = c_con.mean(axis=1)

slope, intercept = np.polyfit(x, y, 1)

plt.figure(figsize=(6,4))
plt.plot(x, y, marker='o', linestyle='None', label=r'$\widehat{c}_{\text{con}}$')
plt.plot(x, slope*np.array(x, dtype='float') + intercept, color="red", 
         label=r"linear fit", linestyle="--")
plt.xlabel(r'$k$')
plt.ylabel(r'$\widehat{c_{\text{con}}}$')
plt.legend()
plt.title(r'$N=$'+str(N)+r', $\sigma=$'+str(sigma)+r', $\lambda=$'+str(lam)+', $s=$'+str(s)+r', target$=\frac{\cos(2kx)+\sin(2kx)}{(2k)^{0.75}}$')
plt.savefig('c_con(k)_target=f2k.png', dpi=600)


# ### target = $\sum_{n=0}^5\cos(nx)$

# In[ ]:


rng = np.random.default_rng(0)
rng_test = np.random.default_rng(42)

N = 10000
sigma = 1.0
lam = 1.0
s = 0.75

k_grid = [i for i in range(0, 11)]
n_iter = 10

mse = np.empty((len(k_grid), n_iter))

print("Done: ", end="")
for ix, k in enumerate(k_grid):
    for i in range(n_iter):
        a = build_coeffs(N, s=s)
        data = make_datasets(N, sigma, k, a, rng, target=target_sumcos)

        model = ConditionalKRR(a=a, k=k, lam_krr=lam).fit(data.X, data.y_signal)

        x_grid = rng.uniform(0.0, 2*np.pi, size=300)
        y_true = target_sumcos(x_grid, k, a)
        y_pred = model.predict(x_grid)

        mse[ix, i] = np.mean((y_pred - y_true)**2)
    print(" ... "+str(round(float(ix+1) / len(k_grid)*100)) + "%", end=" ")



x = k_grid
y = mse.mean(axis=1)

slope, intercept = np.polyfit(x, y, 1)

plt.figure(figsize=(6,4))
plt.plot(x, y, marker='o', linestyle='None', label=r'MSE')
plt.xlabel(r'$k$')
plt.ylabel(r'MSE')
plt.legend()
plt.title(r'$N=$'+str(N)+r', $\sigma=$'+str(round(sigma,2))+r', $\lambda=$'+str(lam)+', $s=$'+str(s)+r', target$=0$')
plt.savefig('mse(k)_target=sumcos.png', dpi=600)


# In[ ]:


x = k_grid
y = mse.mean(axis=1)

slope, intercept = np.polyfit(x, y, 1)

plt.figure(figsize=(6,4))
plt.plot(x, y, marker='o', linestyle='None', label=r'MSE')
plt.xlabel(r'$k$')
plt.ylabel(r'MSE')
plt.legend()
plt.title(r'$N=$'+str(N)+r', $\sigma=$'+str(round(sigma,2))+r', $\lambda=$'+str(lam)+', $s=$'+str(s)+r', target$=0$')
plt.savefig('mse(k)_target=sumcos.png', dpi=600)


# In[ ]:


N = 500
k = 0
a = build_coeffs(N, s=s)
data = make_datasets(N, sigma, k, a, rng, sumcos=True)
model = ConditionalKRR(a=a, k=k, lam_krr=lam).fit(data.X, data.y_signal)

x_grid = np.linspace(0, 2*np.pi, 300, endpoint=False)
y_true = target_sumcos(x_grid, k, a)
y_pred = model.predict(x_grid)

plt.figure(figsize=(6,4))
plt.plot(x_grid, y_true, label=r'$f_'+str(k)+r'(x)$', color='black')
plt.plot(x_grid, y_pred, label=r'$\hat{f}(x)$', linestyle='--', linewidth=.5, color='red')
plt.plot(data.X, data.y_signal, label=r"$f_"+str(k)+r"(x_i)+\epsilon_i$", marker='o', markersize=2, linestyle='None', color='blue')
plt.xlabel(r'$x$')
plt.legend()
plt.title(r'$N=$'+str(N)+r', $\sigma=$'+str(sigma)+r', $\lambda=$'+str(lam)+', $s=$'+str(s)+r', target$=f_'+str(k)+r'$')
plt.savefig('true_vs_pred_N='+str(N)+'_target=sumcos_k='+str(k)+'.png', dpi=600)


# In[ ]:


N = 500
k = 5
a = build_coeffs(N, s=s)
data = make_datasets(N, sigma, k, a, rng, sumcos=True)
model = ConditionalKRR(a=a, k=k, lam_krr=lam).fit(data.X, data.y_signal)

x_grid = np.linspace(0, 2*np.pi, 300, endpoint=False)
y_true = target_sumcos(x_grid, k, a)
y_pred = model.predict(x_grid)

plt.figure(figsize=(6,4))
plt.plot(x_grid, y_true, label=r'$f_{'+str(k)+r'}(x)$', color='black')
plt.plot(x_grid, y_pred, label=r'$\hat{f}(x)$', linestyle='--', linewidth=.5, color='red')
plt.plot(data.X, data.y_signal, label=r"$f_{"+str(k)+r"}(x_i)+\epsilon_i$", marker='o', markersize=2, linestyle='None', color='blue')
plt.xlabel(r'$x$')
plt.legend()
plt.title(r'$N=$'+str(N)+r', $\sigma=$'+str(sigma)+r', $\lambda=$'+str(lam)+', $s=$'+str(s)+r', target$=f_{'+str(k)+r'}$')
plt.savefig('true_vs_pred_N='+str(N)+'_target=sumcos_k='+str(k)+'.png', dpi=600)


# #### target = $\sum\sin(nx)$

# In[ ]:


rng = np.random.default_rng(0)
rng_test = np.random.default_rng(42)

N = 100
sigma = 1.0
lam = 1.0
s = 0.75

k_grid = [i for i in range(0, 11)]
n_iter = 10

mse = np.empty((len(k_grid), n_iter))

print("Done: ", end="")
for ix, k in enumerate(k_grid):
    for i in range(n_iter):
        a = build_coeffs(N, s=s)
        data = make_datasets(N, sigma, k, a, rng, target=target_sumsin)

        model = ConditionalKRR(a=a, k=k, lam_krr=lam).fit(data.X, data.y_signal)

        x_grid = rng.uniform(0.0, 2*np.pi, size=300)
        y_true = target_sumcos(x_grid, k, a)
        y_pred = model.predict(x_grid)

        mse[ix, i] = np.mean((y_pred - y_true)**2)
    print(" ... "+str(round(float(ix+1) / len(k_grid)*100)) + "%", end=" ")



x = k_grid
y = mse.mean(axis=1)

slope, intercept = np.polyfit(x, y, 1)

plt.figure(figsize=(6,4))
plt.plot(x, y, marker='o', linestyle='None', label=r'MSE')
plt.xlabel(r'$k$')
plt.ylabel(r'MSE')
plt.legend()
plt.title(r'$N=$'+str(N)+r', $\sigma=$'+str(round(sigma,2))+r', $\lambda=$'+str(lam)+', $s=$'+str(s)+r', target$=0$')
plt.savefig('mse(k)_target=sumsin.png', dpi=600)


# ## Dependence on $\sigma^2$

# In[74]:


rng = np.random.default_rng(0)
rng_test = np.random.default_rng(42)
k = 5
N = 100
lam = .1
s = .75

sigmasq_grid = [i for i in range(0, 16)]
n_iter = 50

mse = np.empty((len(sigmasq_grid), n_iter))
c_con = np.empty((len(sigmasq_grid), n_iter))

print("Done: ", end="")
for ix, sigmasq in enumerate(sigmasq_grid):
    for i in range(n_iter):
        sigma = np.sqrt(sigmasq)
        a = build_coeffs(N, s=s)
        data = make_datasets(N, sigma, k, a, rng)

        model = ConditionalKRR(a=a, k=k, lam_krr=lam).fit(data.X, data.y_signal)

        x_grid = rng.uniform(0.0, 2*np.pi, size=300)
        y_true = target_fk(x_grid, k, a)
        y_pred = model.predict(x_grid)

        model_noise = NoiseKRR(a=a, k=k, lam=lam).fit(data.X, data.y_noise)
        fhat_noise = model_noise.predict(x_grid)
        true_plus_noise = y_true + fhat_noise

        mse[ix, i] = np.mean((y_pred - y_true)**2)
        c_con[ix, i] = np.mean((y_pred - true_plus_noise)**2)
    print(" ... "+str(round(float(ix+1) / len(sigmasq_grid)*100)) + "%", end=" ")


# In[76]:


x = sigmasq_grid
y = c_con.mean(axis=1)

slope, intercept = np.polyfit(x, y, 1)

plt.figure(figsize=(6,4))
plt.plot(x, y, marker='o', linestyle='None')
plt.plot(x, slope*np.array(x, dtype='float') + intercept, color="red", 
         label=r"linear fit", linestyle="--")
plt.xlabel(r'$\sigma^2$')
plt.ylabel(r'$\widehat{c_\text{con}}$')
plt.legend()
plt.title(r'$N=$'+str(N)+r', $k=$'+str(k)+r', $\lambda=$'+str(lam)+', $s=$'+str(s)+r', target$=\frac{\cos(kx)+\sin(kx)}{k^{0.75}}$')
plt.savefig('c_con(sigmasq)_target=fk.png', dpi=600)


# In[ ]:




