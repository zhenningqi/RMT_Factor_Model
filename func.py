import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import jensenshannon


# ----------------------
# 1. 数据生成 (synthetic data)
# ----------------------
def generate_synthetic_data(
    N, T, p, signal_strength=1.0, noise_level=1.0, rho=0.0, beta=0.0, J=0.0
):
    F = np.random.normal(0, signal_strength, size=(p, T))
    L = np.random.normal(0, 1, size=(N, p))
    common_component = L @ F

    E = np.zeros((N, T))
    V = np.random.normal(0, 1, size=(N, T))
    for t in range(T):
        for i in range(N):
            local_corr = np.sum(
                [beta * V[h, t] for h in range(max(0, i - J), min(N, i + J)) if h != i]
            )
            if t == 0:
                E[i, t] = V[i, t] + local_corr
            else:
                E[i, t] = rho * E[i, t - 1] + V[i, t] + local_corr

    U = np.sqrt((1 - rho**2) / (1 + 2 * J * beta**2)) * E
    X = common_component + np.sqrt(noise_level) * U
    return X


# ----------------------
# 2. PCA残差计算
# ----------------------
def get_residuals(X, p):
    if p > 0:
        pca = PCA(n_components=p)
        X_reduced = pca.fit_transform(X.T).T
        X_reconstructed = pca.components_.T @ X_reduced
        residuals = X - X_reconstructed
        return residuals
    else:
        return X.copy()


# ----------------------
# 3. 经验谱密度
# ----------------------
def empirical_spectral_density(U, num=5, epsilon=1e-2):
    cov = U @ U.T / U.shape[1]
    eigvals = np.linalg.eigvalsh(cov)
    max_val = max(1, np.max(eigvals)) * 1.2
    min_val = epsilon
    range = (min_val, max_val)
    if num is None:
        density, grid = np.histogram(eigvals, bins="auto", range=range, density=True)
    else:
        bins = (max_val - min_val) * num
        bins = int(bins)
        density, grid = np.histogram(eigvals, bins=bins, range=range, density=True)
    centers = 0.5 * (grid[:-1] + grid[1:])
    return centers, density, grid


def draw_bar(ax, centers, density, color="blue", label="Empirical Density"):
    width = (centers[1] - centers[0]) * 0.8
    ax.bar(
        centers,
        density,
        width=width,
        color=color,
        alpha=1,
        label=label,
    )


def calculate_MP_bounds(N, T, theta=1):
    r = N / T
    lambda_min = theta * ((1 - np.sqrt(r)) ** 2)
    lambda_max = theta * ((1 + np.sqrt(r)) ** 2)
    return lambda_min, lambda_max


def MP_density(N, T, theta=1, num=50, epsilon=1e-2):
    r = N / T
    lambda_min, lambda_max = calculate_MP_bounds(N, T, theta=theta)
    range = (max(epsilon, lambda_min), lambda_max)
    bins = max(1, (range[1] - range[0])) * num
    bins = int(bins)
    x = np.linspace(range[0], range[1], bins)
    density = np.sqrt((lambda_max - x) * (x - lambda_min)) / (2 * np.pi * r * x * theta)
    return x, density


def draw_MP(ax, N, T, theta=1, color="red", num=50):
    x, density = MP_density(N, T, theta=theta, num=num)
    ax.plot(x, density, color=color, label="MP-law", lw=2)


# ----------------------
# 4. 理论谱密度
# ----------------------
def MB(z, b):
    numerator = -1
    term1 = np.sqrt(1 - z)
    term2 = np.sqrt(1 - ((1 + b**2) ** 2 / (1 - b**2)) * z)
    return numerator / (term1 * term2)


def subdivide_grid(grid, num):
    result = []
    for i in range(len(grid) - 1):
        segment = np.linspace(grid[i], grid[i + 1], num, endpoint=False)
        result.extend(segment)
    result.append(grid[-1])
    return np.array(result)


def theoretical_spectral_density(b, c, grid, num=1, epsilon=1e-8):
    a2 = 1 - b**2
    if num == 1:
        lam = 0.5 * (grid[:-1] + grid[1:])
        density = np.zeros_like(lam)
        for idx, l in enumerate(lam):
            z = l + 1j * epsilon
            coeffs = [
                (a2**2) * (c**2),
                2 * a2 * c * (-(1 + b**2) * z + a2 * c),
                (1 - b**2) ** 2 * z**2
                - 2 * a2 * c * (1 + b**2) * z
                + (c**2 - 1) * a2**2,
                -2 * a2**2,
                -(a2**2),
            ]
            roots = np.roots(coeffs)
            roots = roots[np.isfinite(roots)]
            M = roots
            G = (M + 1) / z
            val = (-1 / np.pi) * np.imag(G)
            density[idx] = np.max(val)  # NOTE: pick the maximum value
        return lam, density
    else:
        lam = subdivide_grid(grid, num)
        density = np.zeros_like(lam)
        for idx, l in enumerate(lam):
            z = l + 1j * epsilon
            coeffs = [
                (a2**2) * (c**2),
                2 * a2 * c * (-(1 + b**2) * z + a2 * c),
                (1 - b**2) ** 2 * z**2
                - 2 * a2 * c * (1 + b**2) * z
                + (c**2 - 1) * a2**2,
                -2 * a2**2,
                -(a2**2),
            ]
            roots = np.roots(coeffs)
            roots = roots[np.isfinite(roots)]
            M = roots
            G = (M + 1) / z
            val = (-1 / np.pi) * np.imag(G)
            density[idx] = np.max(val)  # NOTE: pick the maximum value
        centers = 0.5 * (grid[:-1] + grid[1:])
        centers_density = np.zeros_like(centers)
        for i in range(len(centers)):
            centers_density[i] += density[i * num]
            centers_density[i] += density[(i + 1) * num]
            for j in range(1, num):
                centers_density[i] += 2 * density[i * num + j]
            centers_density[i] /= 2 * num
        return centers, centers_density


# ----------------------
# 5. Jensen-Shannon 散度计算
# ----------------------
import numpy as np


def smooth_distribution(P, epsilon=1e-12):
    zeros = P == 0
    n_zeros = np.sum(zeros)
    alpha = 1.0 - n_zeros * epsilon
    P_smooth = np.where(P > 0, alpha * P, epsilon)
    return P_smooth


def kl_divergence(P, Q):
    return np.sum(P * np.log2(P / Q))


def js_divergence(P, Q, epsilon=1e-12):
    P = smooth_distribution(P, epsilon)
    Q = smooth_distribution(Q, epsilon)
    M = 0.5 * (P + Q)
    return 0.5 * kl_divergence(P, M) + 0.5 * kl_divergence(Q, M)


def spectral_distance(empirical_density, theoretical_density, grid, mode=0):
    empirical_discrete_prob = empirical_density * (grid[1:] - grid[:-1])
    theoretical_discrete_prob = theoretical_density * (grid[1:] - grid[:-1])
    if np.sum(theoretical_discrete_prob) > 0.95:
        theoretical_discrete_prob /= np.sum(theoretical_discrete_prob)
    else:
        empirical_discrete_prob = np.append(empirical_discrete_prob, 0)
        theoretical_discrete_prob = np.append(
            theoretical_discrete_prob, 1 - theoretical_discrete_prob.sum()
        )
    if mode == 0:
        return js_divergence(empirical_discrete_prob, theoretical_discrete_prob)
    elif mode == 1:
        return jensenshannon(empirical_discrete_prob, theoretical_discrete_prob)


# ----------------------
# 6. 估计p和b
# ----------------------
def estimate_p_and_b(
    X, max_p=10, b_grid=np.linspace(0, 0.99, 100), num=5, subdivide_num=2
):
    N, T = X.shape
    c = N / T

    best_loss = np.inf
    best_p, best_b = None, None
    losses = []

    for p in range(max_p + 1):
        U = get_residuals(X, p)
        _, empirical_density, grid = empirical_spectral_density(U, num=num)
        for b in b_grid:
            _, theoretical_density = theoretical_spectral_density(
                b, c, grid, num=subdivide_num
            )
            loss = spectral_distance(
                empirical_density, theoretical_density, grid, mode=0
            )
            losses.append((p, b, loss))
            if loss < best_loss:
                best_loss, best_p, best_b = loss, p, b

    return best_p, best_b, losses
