import numpy as np
import matplotlib.pyplot as plt
from func import generate_synthetic_data
from func import estimate_p_and_b
from func import get_residuals
from func import empirical_spectral_density
from func import theoretical_spectral_density
from func import draw_bar
from func import draw_MP
from func import subdivide_grid

np.random.seed(345)

true_p = 3
N, T = 400, 1000
rho, beta = 0.5, 0.1

X = generate_synthetic_data(
    N,
    T,
    p=true_p,
    signal_strength=1.0,
    noise_level=1.0,
    rho=rho,
    beta=beta,
    J=N // 50,
)
est_p, est_b, losses = estimate_p_and_b(X, max_p=20, num=8, subdivide_num=2)
print(f"True p: {true_p}, Estimated p: {est_p}, Estimated b: {est_b:.3f}")

U = get_residuals(X, est_p)
empirical_center, empirical_density, grid = empirical_spectral_density(U, num=8)
theoretical_center, theoretical_density = theoretical_spectral_density(
    est_b, N / T, subdivide_grid(grid, num=10)
)

fig, ax = plt.subplots(figsize=(8, 6))
draw_bar(
    ax, empirical_center, empirical_density, label=f"Empirical Density (p={est_p})"
)
draw_MP(ax, N, T, theta=1)
ax.plot(
    theoretical_center,
    theoretical_density,
    label=f"Theoretical Density (b={est_b:.2f})",
    lw=2,
    color="darkorange",
)
ax.set_xlabel("Eigenvalue")
ax.set_ylabel("Density")
ax.legend()
fig.savefig("fig/test3.png", dpi=500)
