import numpy as np
import matplotlib.pyplot as plt
from func import generate_synthetic_data
from func import get_residuals
from func import empirical_spectral_density
from func import draw_bar
from func import draw_MP

np.random.seed(1234)

N, T = 400, 1000
true_p = 3

X = generate_synthetic_data(
    N,
    T,
    p=true_p,
    signal_strength=1.0,
    noise_level=1.0,
    rho=0.0,
    beta=0.0,
    J=N // 10,
)

fig, axes = plt.subplots(3, 2, figsize=(8, 9))
axs = axes.flatten()
for i in range(6):
    U = get_residuals(X, i)
    centers, density, _ = empirical_spectral_density(U, num=8)
    draw_bar(axs[i], centers, density, label="Synthetic Data")
    draw_MP(axs[i], N, T, theta=1)
    axs[i].set_title(f"{i} factor removed")
    axs[i].set_xlabel("Eigenvalue")
    axs[i].set_ylabel("Density")
    axs[i].legend()
fig.tight_layout()
fig.savefig("fig/Example4.png", dpi=500)
